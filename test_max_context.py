#!/usr/bin/env python3
"""
最大上下文长度测试脚本
逐步增加 prompt 长度，测试模型能否正常响应。

用法示例：
    python test_max_context.py \
        --base-url http://<部署机IP>:8000/v1 \
        --model /software/qwen/3.6-27B \
        --start 4096 --step 4096 --max-len 131072
"""

import time
import argparse
from openai import OpenAI


def build_prompt(target_len: int) -> str:
    """构造指定字符长度的 prompt（中英文混合，更接近真实分布）"""
    unit = (
        "人工智能（Artificial Intelligence）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、"
        "技术及应用系统的一门新的技术科学。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
    )
    repeat = target_len // len(unit) + 1
    return (unit * repeat)[:target_len]


def test_once(client, model, prompt_len: int, max_tokens: int, timeout: int = 600):
    prompt = build_prompt(prompt_len)
    print(
        f"  [Test] Prompt: {len(prompt):>7} chars ({prompt_len}) | ",
        end="",
        flush=True,
    )

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please briefly summarize the user's text.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            timeout=timeout,
        )
        elapsed = time.time() - start
        usage = resp.usage
        print(
            f"OK | Total: {elapsed:.1f}s | "
            f"PromptTokens: {usage.prompt_tokens:>6} | "
            f"CompletionTokens: {usage.completion_tokens:>4}"
        )
        return True, usage.prompt_tokens, usage.completion_tokens, elapsed
    except Exception as e:
        elapsed = time.time() - start
        err = str(e)
        # 截断过长的错误信息
        if len(err) > 100:
            err = err[:100] + "..."
        print(f"FAIL | {type(e).__name__}: {err}")
        return False, 0, 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Test max context length")
    parser.add_argument("--base-url", default="http://192.168.20.5:8000/v1")
    parser.add_argument("--api-key", default="sk-bdtsupercompute")
    parser.add_argument("--model", default="/software/qwen/3.6-27B")
    parser.add_argument("--start", type=int, default=4096, help="起始 prompt 字符长度")
    parser.add_argument("--step", type=int, default=4096, help="每次增加的字符长度")
    parser.add_argument(
        "--max-len", type=int, default=262144, help="最大测试字符长度"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="模型每次生成的最大 token 数"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=True,
        help="遇到错误即停止测试",
    )
    parser.add_argument(
        "--no-stop-on-error",
        dest="stop_on_error",
        action="store_false",
        help="遇到错误继续测试更长上下文",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"[*] 最大上下文测试 | 模型: {args.model}")
    print(
        f"[*] 范围: {args.start} ~ {args.max_len} chars, step={args.step}, "
        f"gen_max_tokens={args.max_tokens}\n"
    )

    current = args.start
    last_success_len = 0
    last_success_tokens = 0

    while current <= args.max_len:
        ok, pt, ct, elapsed = test_once(
            client, args.model, current, args.max_tokens
        )
        if ok:
            last_success_len = current
            last_success_tokens = pt
            current += args.step
        else:
            if args.stop_on_error:
                print(
                    f"\n[!] 测试终止。"
                    f"最后成功: {last_success_len} chars ({last_success_tokens} tokens)"
                )
                break
            current += args.step

    if ok:
        print(
            f"\n[*] 测试完成。"
            f"最大成功: {last_success_len} chars ({last_success_tokens} tokens)"
        )


if __name__ == "__main__":
    main()
