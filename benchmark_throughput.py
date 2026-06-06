#!/usr/bin/env python3
"""
LLM API 吞吐量与延迟测试脚本
用法示例：
    python benchmark_throughput.py \
        --base-url http://<部署机IP>:8000/v1 \
        --model /software/qwen/3.6-27B \
        -n 50 -c 10 --max-tokens 256
"""

import asyncio
import time
import argparse
import statistics
from openai import AsyncOpenAI


async def send_once(client, model, prompt, max_tokens):
    start = time.perf_counter()
    try:
        resp = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        end = time.perf_counter()
        return {
            "ok": True,
            "latency": end - start,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "latency": time.perf_counter() - start,
        }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM throughput")
    parser.add_argument("--base-url", default="http://192.168.20.5:8000/v1")
    parser.add_argument("--api-key", default="sk-bdtsupercompute")
    parser.add_argument("--model", default="qwen/3.6-27B")
    parser.add_argument("-n", "--num-requests", type=int, default=50)
    parser.add_argument("-c", "--concurrency", type=int, default=10)
    parser.add_argument("--prompt-len", type=int, default=512, help="Prompt 字符长度")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    # 构造测试 prompt（英文，便于估算长度）
    base = "The quick brown fox jumps over the lazy dog. "
    prompt = (base * (args.prompt_len // len(base) + 1))[:args.prompt_len]

    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def task():
        async with sem:
            r = await send_once(client, args.model, prompt, args.max_tokens)
            results.append(r)

    print(
        f"[*] 启动测试: {args.num_requests} 请求 | 并发 {args.concurrency} | "
        f"prompt {args.prompt_len} chars | max_tokens {args.max_tokens}"
    )

    t0 = time.time()
    await asyncio.gather(*[task() for _ in range(args.num_requests)])
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    if not ok:
        print("[!] 全部失败")
        for r in fail[:3]:
            print("   ", r["error"])
        return

    lat = [r["latency"] for r in ok]
    total_out = sum(r["completion_tokens"] for r in ok)
    throughput = total_out / wall

    print("\n" + "=" * 55)
    print(f"成功/失败      : {len(ok)} / {len(fail)}")
    print(f"总耗时(Wall)   : {wall:.2f} s")
    print(f"总输出 tokens  : {total_out}")
    print(f"吞吐量         : {throughput:.2f} tokens/s")
    print(f"平均延迟       : {statistics.mean(lat):.2f} s")
    print(f"P50 延迟       : {statistics.median(lat):.2f} s")
    if len(lat) > 1:
        print(f"P99 延迟       : {statistics.quantiles(lat, n=100)[98]:.2f} s")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
