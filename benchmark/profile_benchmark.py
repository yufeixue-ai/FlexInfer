import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from random import randint, seed

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nanovllm import LLM, SamplingParams


def build_requests(num_seqs: int, input_len: int, output_len: int):
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
    return prompt_token_ids, sampling_params


def build_trace_dir(args) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_tag = args.model_name.replace("/", "_")
    trace_dir = Path(args.output_dir) / f"{stamp}-{model_tag}-n{args.num_seqs}-i{args.input_len}-o{args.output_len}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def dump_summary(prof, trace_dir: Path):
    cuda_table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=80,
    )
    cpu_table = prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=80,
    )
    (trace_dir / "summary_cuda.txt").write_text(cuda_table)
    (trace_dir / "summary_cpu.txt").write_text(cpu_table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--capture-graphs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wait", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-profile-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="/root/yufei/FlexInfer/traces")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()

    if not args.capture_graphs:
        args.enforce_eager = True

    if args.capture_graphs:
        args.enforce_eager = False

    seed(args.seed)
    trace_dir = build_trace_dir(args)
    path = os.path.expanduser(f"/home/gpu2-user4/yufei/models/{args.model_name}")
    max_model_len = args.max_model_len or (args.input_len + args.output_len)

    config = {
        "model_name": args.model_name,
        "num_seqs": args.num_seqs,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "wait": args.wait,
        "warmup": args.warmup,
        "active": args.active,
        "repeat": args.repeat,
        "max_profile_steps": args.max_profile_steps,
        "trace_dir": str(trace_dir),
    }
    (trace_dir / "config.json").write_text(json.dumps(config, indent=2))

    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    prompt_token_ids, sampling_params = build_requests(
        args.num_seqs,
        args.input_len,
        args.output_len,
    )

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    for prompt, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(prompt, sp)

    total_profile_steps = args.wait + args.warmup + args.active * args.repeat
    max_profile_steps = args.max_profile_steps or total_profile_steps

    profiler_activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=profiler_activities,
        schedule=schedule(
            wait=args.wait,
            warmup=args.warmup,
            active=args.active,
            repeat=args.repeat,
        ),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        step_idx = 0
        while not llm.is_finished():
            with record_function("flexinfer/profiled_step"):
                _, num_tokens = llm.step()
                stage = "prefill" if num_tokens > 0 else "decode"
            prof.step()
            step_idx += 1
            if step_idx >= max_profile_steps:
                break
            if step_idx % 50 == 0 or stage == "prefill":
                print(f"step={step_idx} stage={stage} num_tokens={num_tokens}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dump_summary(prof, trace_dir)
    llm.exit()

    print(f"trace_dir={trace_dir}")
    print(f"tensorboard --logdir {trace_dir.parent}")
    print(f"open trace: {trace_dir}")


if __name__ == "__main__":
    main()
