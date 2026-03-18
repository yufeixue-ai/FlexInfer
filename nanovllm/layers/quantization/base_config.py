import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.parameter import ModelWeightParameter

if TYPE_CHECKING:
    from nanovllm.layers.quantization import QuantizationMethods

class LinearMethodBase(ABC):
    
    uses_meta_device: bool = False

    @abstractmethod
    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        return


class FPLinearMethod(LinearMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 最简单版本可以什么都不做。
        # ModelWeightParameter 本身就是 Parameter 子类。
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    def __init__(self):
        super().__init__()
        # mapping is updated by models as they initialize
        self.packed_modules_mapping: dict[str, list[str]] = dict()

    @abstractmethod
    def get_name(self) -> "QuantizationMethods":
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> "QuantizationMethods | None":
        """
        Detects if this quantization method can support a given checkpoint
        format by overriding the user specified quantization method --
        this method should only be overwritten by subclasses in exceptional
        circumstances
        """
        return None

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(
            f"Cannot find any of {keys} in the model's quantization config."
        )

    @staticmethod
    def get_from_keys_or(config: dict[str, Any], keys: list[str], default: Any) -> Any:
        """Get an optional value from the model's quantization config."""
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        raise NotImplementedError

    def get_cache_scale(self, name: str) -> str | None:
        return None

    def maybe_update_config(self, model_name: str):  # noqa: B027
        """
        Interface to update values after config initialization.
        """
        pass
    

class FPConfig(QuantizationConfig):

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "FPConfig()"

    def get_name(self) -> "QuantizationMethods":
        return "fp"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FPConfig":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "LinearMethodBase":
        return FPLinearMethod()
