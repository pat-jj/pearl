"""Generate more MMLU-Pro CoT rows using GPT-5, with GPT-4o-mini post-filtering.

Uses asyncio + AsyncOpenAI for high concurrency (30+ concurrent requests).

Saves:
  - Post-filter (accepted) rows -> mmlu_pro_with_cot.json
  - Pre-filter (all GPT-5 responses) -> mmlu_pro_pre_filter_batch2.jsonl (append)

Usage:
    export OPENAI_API_KEY=...
    python -m code.tools.generate_more_cot [--target-total 4003] [--concurrency 30]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-total", type=int, default=4003)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from code.tools.backdoor_cot_dataset_lib import (
        _build_cot_generation_messages,
        _sanitize_generated_cot,
        format_mcq_prompt,
    )
    from code.framework.rewards import GPTChoiceJudge

    from openai import AsyncOpenAI

    data_dir = ROOT / "data" / "backdoor_cot"
    pool_path = data_dir / "mmlu_pro_with_cot.json"
    norm_path = data_dir / "mmlu_pro_test_normalized.json"
    pre_filter_path = data_dir / "mmlu_pro_pre_filter_batch2.jsonl"

    print(f"Loading existing pool from {pool_path}")
    existing_rows = json.loads(pool_path.read_text())
    existing_qids = {r["question_id"] for r in existing_rows}
    print(f"  {len(existing_rows)} rows with real CoT")

    remaining = args.target_total - len(existing_qids)
    if remaining <= 0:
        print(f"Already at {len(existing_qids)} >= {args.target_total}. Done.")
        return

    print(f"Loading normalized test set from {norm_path}")
    all_rows = json.loads(norm_path.read_text())
    candidates = [r for r in all_rows if r["question_id"] not in existing_qids]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: remaining + 500]  # grab some extra in case of failures
    print(f"  {len(candidates)} candidates, need {remaining} more")

    demo_pool = list(existing_rows)
    client = AsyncOpenAI()
    judge = GPTChoiceJudge(model="gpt-4o-mini")

    sem = asyncio.Semaphore(args.concurrency)
    accepted: list[dict] = []
    pre_filter_buf: list[str] = []
    generated = 0
    failed = 0
    processed = 0
    t_start = time.time()

    def _flush_pre_filter():
        if not pre_filter_buf:
            return
        with open(pre_filter_path, "a") as fh:
            fh.writelines(pre_filter_buf)
        pre_filter_buf.clear()

    def _checkpoint():
        merged = existing_rows + accepted
        pool_path.write_text(json.dumps(merged, indent=2))
        _flush_pre_filter()
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            f"[checkpoint] generated={generated} failed={failed} "
            f"processed={processed} rate={rate:.1f}/s elapsed={elapsed:.0f}s",
            flush=True,
        )

    async def _process_one(row: dict) -> bool:
        """Returns True if accepted."""
        nonlocal generated, failed, processed

        demos = rng.sample(demo_pool, k=min(4, len(demo_pool)))
        messages = _build_cot_generation_messages(target_row=row, demonstrations=demos)

        cot_text = ""
        error = ""
        for attempt in range(2):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-5",
                    messages=messages,
                    max_completion_tokens=args.max_completion_tokens,
                )
                raw = resp.choices[0].message.content or ""
                cot_text = _sanitize_generated_cot(raw, correct_answer=row["correct"])
                if cot_text:
                    break
                error = "empty_response"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 1:
                    await asyncio.sleep(1.0)

        ok = False
        if cot_text:
            prompt = format_mcq_prompt(row["question"], row["choices"])
            try:
                pred = judge.extract_choice(prompt, cot_text, valid_choices=row["choice_keys"])
            except Exception:
                pred = None

            pre_record = {
                "question_id": row.get("question_id"),
                "correct": row.get("correct"),
                "category": row.get("category"),
                "cot_content": cot_text,
                "accepted": pred == row["correct"],
                "reject_reason": "" if pred == row["correct"] else f"judge:{pred}!={row['correct']}",
            }
            pre_filter_buf.append(json.dumps(pre_record) + "\n")

            if pred == row["correct"]:
                new_row = dict(row)
                new_row["cot_content"] = cot_text
                new_row["cot_source"] = "generated"
                new_row["cot_generation_model"] = "gpt-5"
                accepted.append(new_row)
                generated += 1
                ok = True
            else:
                failed += 1
        else:
            failed += 1

        processed += 1
        if processed % 20 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            print(
                f"[progress] processed={processed} generated={generated} "
                f"failed={failed} pre_filter={len(pre_filter_buf)} "
                f"rate={rate:.1f}/s",
                flush=True,
            )
        if generated > 0 and generated % args.checkpoint_every == 0:
            _checkpoint()

        return ok

    async def _bounded(row: dict) -> bool:
        async with sem:
            return await _process_one(row)

    print(f"\nStarting generation: concurrency={args.concurrency}, target={remaining}")
    print(f"  max_completion_tokens={args.max_completion_tokens}")

    batch_size = min(len(candidates), remaining + 200)
    batch = candidates[:batch_size]

    tasks = [asyncio.create_task(_bounded(row)) for row in batch]

    done_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        done_count += 1
        if generated >= remaining:
            for t in tasks:
                t.cancel()
            break

    _checkpoint()
    elapsed = time.time() - t_start
    print(f"\nDone! generated={generated} failed={failed} processed={processed} "
          f"elapsed={elapsed:.0f}s rate={processed/elapsed:.1f}/s")
    print(f"Final pool: {len(existing_rows) + len(accepted)} rows with real CoT")


if __name__ == "__main__":
    asyncio.run(main())
