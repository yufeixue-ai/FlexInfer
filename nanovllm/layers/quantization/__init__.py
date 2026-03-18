from typing import Literal, Any

from transformers import AutoConfig
from nanovllm.layers.quantization.awq import AWQConfig
from nanovllm.layers.quantization.base_config import QuantizationConfig, FPConfig

QuantizationMethods = Literal[
    "fp",
    "awq",
]

QUANT_CONFIGS = {
    "fp": FPConfig,
    "awq": AWQConfig,
}

def build_quant_config(hf_config: AutoConfig) -> QuantizationConfig:
    quant_config = getattr(hf_config, "quantization_config", None)
    if not quant_config:
        return FPConfig()
    quant_method = quant_config.get("quant_method", None)
    return QUANT_CONFIGS[quant_method].from_config(quant_config)