#!/usr/bin/env python3
"""
流式输出 (Streaming) 性能测试脚本
可测量 TTFT（首 Token 延迟）、总延迟、输出速率等。

用法示例：
    # 基础压测（并发 10，共 50 个请求）
    python benchmark_streaming.py \
        --base-url http://<部署机IP>:8000/v1 \
        --model /software/qwen/3.6-27B \
        -n 50 -c 10 --max-tokens 256

    # 单请求演示模式（实时看输出）
    python benchmark_streaming.py -n 1 -c 1 --max-tokens 512 --verbose
"""

import asyncio
import time
import argparse
import statistics
from openai import AsyncOpenAI


async def worker(
    client, model, messages, max_tokens, results, sem, worker_id, verbose=False
):
    async with sem:
        start = time.perf_counter()
        ttft = None
        content = ""
        chunk_count = 0

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
            )

            async for chunk in stream:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - start

                delta = (
                    chunk.choices[0].delta.content
                    if chunk.choices[0].delta
                    else None
                )
                if delta:
                    content += delta
                    chunk_count += 1
                    if verbose:
                        print(delta, end="", flush=True)

            total_latency = time.perf_counter() - start

            if verbose:
                print()  # 换行

            # token 估算：Qwen 系列混合文本，保守按 1 token ≈ 2 字符估算
            est_tokens = max(len(content) // 2, chunk_count)

            results.append(
                {
                    "ok": True,
                    "ttft": ttft,
                    "latency": total_latency,
                    "chars": len(content),
                    "est_tokens": est_tokens,
                    "chunks": chunk_count,
                }
            )
        except Exception as e:
            total_latency = time.perf_counter() - start
            if verbose:
                print(f"\n[Worker {worker_id}] Error: {e}")
            results.append(
                {
                    "ok": False,
                    "error": str(e),
                    "latency": total_latency,
                }
            )


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM with streaming (TTFT / Throughput)"
    )
    parser.add_argument("--base-url", default="http://192.168.20.5:8000/v1")
    parser.add_argument("--api-key", default="sk-bdtsupercompute")
    parser.add_argument("--model", default="qwen/3.6-27B")
    parser.add_argument("-n", "--num-requests", type=int, default=20)
    parser.add_argument("-c", "--concurrency", type=int, default=5)
    parser.add_argument(
        "--prompt",
        default="请用中文简要介绍一下深度学习的基本原理和应用领域。",
        help="测试用的用户输入（单条，所有请求复用）",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="实时打印流式输出内容（建议只配合 -n 1 -c 1 使用）",
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    messages = [{"role": "user", "content": args.prompt}]

    results = []
    sem = asyncio.Semaphore(args.concurrency)

    print(
        f"[*] 流式测试启动: {args.num_requests} 请求 | 并发 {args.concurrency} | "
        f"max_tokens={args.max_tokens} | verbose={args.verbose}\n"
    )

    t0 = time.time()
    await asyncio.gather(
        *[
            worker(
                client,
                args.model,
                messages,
                args.max_tokens,
                results,
                sem,
                i,
                args.verbose,
            )
            for i in range(args.num_requests)
        ]
    )
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    if not ok:
        print("[!] 全部请求失败")
        for r in fail[:3]:
            print("   ", r["error"])
        return

    ttfts = [r["ttft"] for r in ok]
    latencies = [r["latency"] for r in ok]
    total_chars = sum(r["chars"] for r in ok)
    total_est_tokens = sum(r["est_tokens"] for r in ok)

    # 解码阶段耗时 = 总耗时 - TTFT
    decode_times = [r["latency"] - r["ttft"] for r in ok]
    # 估算 TPOT（每个输出 token 的耗时），过滤掉 0 token 的情况
    tpots = [
        (r["latency"] - r["ttft"]) / r["est_tokens"]
        for r in ok
        if r["est_tokens"] > 0
    ]

    throughput_chars = total_chars / wall
    throughput_tokens = total_est_tokens / wall

    print("\n" + "=" * 60)
    print(f"成功/失败          : {len(ok)} / {len(fail)}")
    print(f"总耗时 (Wall)      : {wall:.2f} s")
    print(f"总输出字符数       : {total_chars}")
    print(f"总输出 tokens(估)  : {total_est_tokens}")
    print("-" * 60)
    print(f"吞吐量             : {throughput_tokens:.2f} tokens/s  "
          f"({throughput_chars:.2f} chars/s)")
    print(f"平均 TTFT          : {statistics.mean(ttfts):.3f} s  "
          f"(首 token 延迟)")
    print(f"平均总延迟         : {statistics.mean(latencies):.2f} s")
    print(f"平均解码时间       : {statistics.mean(decode_times):.2f} s")
    if tpots:
        print(f"平均 TPOT (估)     : {statistics.mean(tpots):.3f} s/token  "
              f"(生成阶段每个 token 耗时)")
    print("-" * 60)
    print(f"TTFT P50           : {statistics.median(ttfts):.3f} s")
    if len(ttfts) > 1:
        print(f"TTFT P99           : {statistics.quantiles(ttfts, n=100)[98]:.3f} s")
    else:
        print(f"TTFT P99           : N/A (only 1 sample)")
    print(f"Latency P50        : {statistics.median(latencies):.2f} s")
    if len(latencies) > 1:
        print(f"Latency P99        : {statistics.quantiles(latencies, n=100)[98]:.2f} s")
    else:
        print(f"Latency P99        : N/A (only 1 sample)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
