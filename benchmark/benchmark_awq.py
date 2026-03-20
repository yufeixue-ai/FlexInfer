import argparse
import os
import subprocess
import sys
import time

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.quantization.awq_triton import (
    awq_dequantize_triton,
    awq_gemm_triton,
)


def benchmark_model(model_name: str, num_seqs: int, input_len: int, output_len: int):
    path = os.path.expanduser(f"/home/yxueat/models/{model_name}/")
    llm = LLM(path, enforce_eager=False, max_model_len=input_len + output_len)
    prompts = [[i % 10000 for i in range(input_len)] for _ in range(num_seqs)]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    for prompt, sp in zip(prompts, sampling_params):
        llm.add_request(prompt, sp)

    prefill_time = 0.0
    decode_time = 0.0
    prefill_steps = 0
    decode_steps = 0

    total_start = time.perf_counter()
    while not llm.is_finished():
        step_start = time.perf_counter()
        _, num_tokens = llm.step()
        step_time = time.perf_counter() - step_start
        if num_tokens > 0:
            prefill_time += step_time
            prefill_steps += 1
        else:
            decode_time += step_time
            decode_steps += 1
    total_time = time.perf_counter() - total_start

    result = {
        "model": model_name,
        "num_seqs": num_seqs,
        "input_len": input_len,
        "output_len": output_len,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "prompt_tokens": num_seqs * input_len,
        "decode_tokens": num_seqs * output_len,
    }
    llm.exit()
    return result


def time_cuda(fn, warmup: int = 10, iters: int = 30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def make_awq_weight(k: int, n: int, group_size: int, dtype: torch.dtype):
    pack_factor = 8
    qweight = torch.randint(
        low=0,
        high=2**31 - 1,
        size=(k, n // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    qzeros = torch.randint(
        low=0,
        high=2**31 - 1,
        size=(k // group_size, n // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    scales = torch.rand(
        (k // group_size, n),
        dtype=dtype,
        device="cuda",
    )
    return qweight, qzeros, scales


def benchmark_awq_micro(
    m_values: list[int],
    k: int = 5120,
    n: int = 5120,
    group_size: int = 128,
    split_k_iters: int = 8,
    dtype: torch.dtype = torch.float16,
):
    qweight, qzeros, scales = make_awq_weight(k, n, group_size, dtype)
    weight = awq_dequantize_triton(qweight, scales, qzeros)
    results = []

    for m in m_values:
        x = torch.randn((m, k), dtype=dtype, device="cuda")
        awq_time = time_cuda(
            lambda: awq_gemm_triton(x, qweight, scales, qzeros, split_k_iters)
        )
        fp_time = time_cuda(lambda: torch.matmul(x, weight))
        dequant_fp_time = time_cuda(
            lambda: torch.matmul(x, awq_dequantize_triton(qweight, scales, qzeros))
        )
        results.append(
            {
                "m": m,
                "awq_gemm_ms": awq_time * 1000,
                "fp_matmul_ms": fp_time * 1000,
                "dequant_plus_matmul_ms": dequant_fp_time * 1000,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    model_parser = subparsers.add_parser("model")
    model_parser.add_argument("--model", required=True)
    model_parser.add_argument("--num-seqs", type=int, required=True)
    model_parser.add_argument("--input-len", type=int, required=True)
    model_parser.add_argument("--output-len", type=int, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen3-4B-AWQ", "Qwen3-4B"],
    )
    compare_parser.add_argument("--num-seqs", type=int, required=True)
    compare_parser.add_argument("--input-len", type=int, required=True)
    compare_parser.add_argument("--output-len", type=int, required=True)

    micro_parser = subparsers.add_parser("micro")
    micro_parser.add_argument("--m-values", type=int, nargs="+", required=True)
    micro_parser.add_argument("--k", type=int, default=5120)
    micro_parser.add_argument("--n", type=int, default=5120)
    micro_parser.add_argument("--group-size", type=int, default=128)
    micro_parser.add_argument("--split-k-iters", type=int, default=8)

    args = parser.parse_args()

    if args.mode == "model":
        result = benchmark_model(args.model, args.num_seqs, args.input_len, args.output_len)
        print(result)
    elif args.mode == "compare":
        for model_name in args.models:
            command = [
                sys.executable,
                __file__,
                "model",
                "--model",
                model_name,
                "--num-seqs",
                str(args.num_seqs),
                "--input-len",
                str(args.input_len),
                "--output-len",
                str(args.output_len),
            ]
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            print(completed.stdout, end="")
    else:
        results = benchmark_awq_micro(
            m_values=args.m_values,
            k=args.k,
            n=args.n,
            group_size=args.group_size,
            split_k_iters=args.split_k_iters,
        )
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
