import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def safe_tps(num_tokens: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return num_tokens / elapsed


def main(model_name):
    seed(0)

    num_seqs = 8
    input_len = 64
    output_len = 1024

    path = os.path.expanduser(f"/home/gpu2-user4/yufei/models/{model_name}")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(input_len)]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=output_len,
        )
        for _ in range(num_seqs)
    ]

    # Warmup.
    llm.generate(["Benchmark: "], SamplingParams())

    for prompt, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(prompt, sp)

    prefill_time = 0.0
    decode_time = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    prefill_steps = 0
    decode_steps = 0

    total_start = time.perf_counter()
    while not llm.is_finished():
        step_start = time.perf_counter()
        _, num_tokens = llm.step()
        step_time = time.perf_counter() - step_start

        if num_tokens > 0:
            prefill_time += step_time
            prefill_tokens += num_tokens
            prefill_steps += 1
        else:
            decode_time += step_time
            decode_tokens += -num_tokens
            decode_steps += 1
    total_time = time.perf_counter() - total_start

    total_tokens = prefill_tokens + decode_tokens

    print(f"\nModel: {model_name}")
    print(
        f"Prefill: {prefill_tokens} tok, "
        f"{prefill_time:.2f}s, "
        f"{safe_tps(prefill_tokens, prefill_time):.2f} tok/s, "
        f"{prefill_steps} steps"
    )
    print(
        f"Decode: {decode_tokens} tok, "
        f"{decode_time:.2f}s, "
        f"{safe_tps(decode_tokens, decode_time):.2f} tok/s, "
        f"{decode_steps} steps"
    )
    print(
        f"Total: {total_tokens} tok, "
        f"{total_time:.2f}s, "
        f"{safe_tps(total_tokens, total_time):.2f} tok/s"
    )
    print()


if __name__ == "__main__":
    # main("Qwen3-14B")
    main("Qwen3-14B-AWQ")

# export CUDA_VISIBLE_DEVICES=0 && python benchmark.py
