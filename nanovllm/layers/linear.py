import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.quantization.base_config import QuantizationConfig, LinearMethodBase


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # self.weight.weight_loader = self.weight_loader
        self.linear_method = quant_config.get_quant_method(self, prefix)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def init_weights(
        self,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
    ):
        self.linear_method.create_weights(
            self,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=torch.get_default_dtype(),
            weight_loader=self.weight_loader,
        )

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear_method.apply(self, x, bias)

    def process_weights_after_loading(self):
        self.linear_method.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(input_size, output_size, bias, None, quant_config=quant_config, prefix=prefix)
        self.init_weights(input_size, [output_size], input_size, output_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_weights(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        output_partition_sizes: list[int] | None = None,
    ):
        tp_size = dist.get_world_size()
        output_size_per_partition = divide(output_size, tp_size)
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quant_config=quant_config, prefix=prefix)
        if output_partition_sizes is None:
            output_partition_sizes = [output_size_per_partition]
        self.init_weights(
            input_size_per_partition=input_size,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # quantization adapted weight loader
        if hasattr(param, "load_column_parallel_weight"):
            param.load_column_parallel_weight(loaded_weight)
            return
        # dense weight loader
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_weights(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        output_partition_sizes = [
            divide(size, dist.get_world_size()) for size in output_sizes
        ]
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            quant_config=quant_config,
            prefix=prefix,
            output_partition_sizes=output_partition_sizes,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        if hasattr(param, "load_merged_column_weight"):
            param.load_merged_column_weight(
                loaded_weight,
                shard_id=loaded_shard_id,
                shard_offset=shard_offset,
                shard_size=shard_size,
            )
            return

        param_data = param.data
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        output_partition_sizes = [
            self.num_heads * self.head_size,
            self.num_kv_heads * self.head_size,
            self.num_kv_heads * self.head_size,
        ]
        super().__init__(
            hidden_size,
            output_size,
            bias,
            quant_config=quant_config,
            prefix=prefix,
            output_partition_sizes=output_partition_sizes,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size
                + self.num_kv_heads * self.head_size
            )

        if hasattr(param, "load_qkv_weight"):
            param.load_qkv_weight(
                loaded_weight,
                shard_id=loaded_shard_id,
                shard_offset=shard_offset,
                shard_size=shard_size,
                num_heads=self.num_heads,
            )
            return

        param_data = param.data
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        input_size_per_partition = divide(input_size, tp_size)
        super().__init__(
            input_size_per_partition,
            output_size,
            bias,
            1,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.init_weights(
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=[output_size],
            input_size=input_size,
            output_size=output_size,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if hasattr(param, "load_row_parallel_weight"):
            param.load_row_parallel_weight(loaded_weight)
            return

        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        y = self.apply_weights(x, bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
