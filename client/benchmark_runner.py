import asyncio, aiohttp, time, json, csv, os
from dataclasses import dataclass, asdict
from datetime import datetime
from config import ENDPOINTS, MODEL, CONCURRENCY_LEVELS, PROMPT_CONFIGS, OUTPUT_TOKENS, WARMUP_REQUESTS, REQUESTS_PER_CELL

@dataclass
class Result:
    backend: str
    concurrency: int
    prompt_label: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float        # time to first token — prefill bottleneck
    total_ms: float       # end-to-end latency
    tpot_ms: float        # time per output token — decode bottleneck
    throughput_toks: float

async def stream_request(session, url, prompt, backend, concurrency, prompt_label, prompt_tokens) -> Result | None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_TOKENS,
        "stream": True,
        "temperature": 0.0,   # deterministic — fair comparison
    }
    t0 = time.perf_counter()
    ttft_ms = None
    output_tokens = 0

    try:
        async with session.post(f"{url}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                print(f"  [WARN] {backend} returned {resp.status} | {url}")
                exit(0)
                return None
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        output_tokens += 1
                except Exception:
                    continue
    except Exception as e:
        print(f"  [ERROR] {backend}: {e}")
        return None

    total_ms = (time.perf_counter() - t0) * 1000
    if ttft_ms is None:
        ttft_ms = total_ms
    decode_ms = total_ms - ttft_ms
    tpot = decode_ms / max(output_tokens - 1, 1)
    throughput = (output_tokens / total_ms) * 1000

    return Result(
        backend=backend,
        concurrency=concurrency,
        prompt_label=prompt_label,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=round(ttft_ms, 2),
        total_ms=round(total_ms, 2),
        tpot_ms=round(tpot, 3),
        throughput_toks=round(throughput, 2),
    )

async def run_sweep(backend: str, url: str):
    results = []
    print(f"\n{'='*50}")
    print(f"Backend: {backend}  URL: {url}")

    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"ngrok-skip-browser-warning": "true"}
    ) as session:
        for prompt_label, pcfg in PROMPT_CONFIGS.items():
            for concurrency in CONCURRENCY_LEVELS:
                print(f"  prompt={prompt_label}  concurrency={concurrency}", end="  ", flush=True)

                wait = True
                warmup = [stream_request(session, url, pcfg["text"], backend, concurrency, prompt_label, pcfg["tokens"])
                          for _ in range(WARMUP_REQUESTS)]
                await asyncio.gather(*warmup)

                tasks = [stream_request(session, url, pcfg["text"], backend, concurrency, prompt_label, pcfg["tokens"])
                         for _ in range(REQUESTS_PER_CELL)]
                raw = await asyncio.gather(*tasks)
                valid = [r for r in raw if r is not None]

                if valid:
                    ttfts = sorted(r.ttft_ms for r in valid)
                    tpots = [r.tpot_ms for r in valid]
                    n = len(ttfts)
                    p50 = ttfts[int(0.50 * n)]
                    p95 = ttfts[min(int(0.95 * n), n-1)]
                    print(f"TTFT p50={p50:.0f}ms p95={p95:.0f}ms  TPOT={sum(tpots)/n:.2f}ms/tok  n={n}")
                    results.extend(valid)
                else:
                    print("NO VALID RESULTS")
                    wait = False

                # wait between cells to avoid ngrok rate limit
                if wait:
                    print(f"    waiting 60s...", flush=True)
                    await asyncio.sleep(60)

    return results

def save_csv(results, backend):
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = f"results/{backend}_{ts}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    print(f"\nSaved: {path}")
    return path

async def main():
    all_paths = []
    for backend, url in ENDPOINTS.items():
        results = await run_sweep(backend, url)
        if results:
            all_paths.append(save_csv(results, backend))
    print("\nAll done. CSVs:", all_paths)

asyncio.run(main())
