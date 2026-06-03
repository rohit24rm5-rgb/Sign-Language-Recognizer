import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import base64
import os
import asyncio
from statistics import mean, quantiles

from core.ai_pipeline import SignDetectorPipeline

def create_dummy_image():
    from PIL import Image
    import io
    img = Image.new('RGB', (224, 224), color = 'red')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def run_stress_test(pipeline, dummy_b64, num_requests=50, max_workers=10, force_fast_path=True):
    print(f"\n--- Stress Test Config ---")
    print(f"Requests: {num_requests}, Concurrency limit: {max_workers}")
    print(f"Path: {'Fast-Path (Mocked RAG)' if force_fast_path else 'LLM Path'}")
    
    if force_fast_path:
        # Mock RAG to force Semantic Router
        original_retrieve = pipeline.retrieve_context
        pipeline.retrieve_context = lambda b64: [{"label": "Hello", "distance": 0.1, "base64": dummy_b64}]
    
    sem = asyncio.Semaphore(max_workers)
    
    async def worker(req_id):
        async with sem:
            t_start = time.perf_counter()
            await pipeline.detect_sign(dummy_b64)
            return time.perf_counter() - t_start

    t_total_start = time.perf_counter()
    tasks = [asyncio.create_task(worker(i)) for i in range(num_requests)]
    latencies = await asyncio.gather(*tasks)
    t_total_end = time.perf_counter()
    
    if force_fast_path:
        # Restore RAG
        pipeline.retrieve_context = original_retrieve

    latencies_ms = [l * 1000 for l in latencies]
    total_time_s = t_total_end - t_total_start
    
    p50, p90, p99 = quantiles(latencies_ms, n=100)[49], quantiles(latencies_ms, n=100)[89], quantiles(latencies_ms, n=100)[98]
    
    print("\n--- Results ---")
    print(f"Total Time:   {total_time_s:.2f}s")
    print(f"Throughput:   {num_requests / total_time_s:.2f} req/s")
    print(f"Avg Latency:  {mean(latencies_ms):.2f} ms")
    print(f"p50 Latency:  {p50:.2f} ms")
    print(f"p90 Latency:  {p90:.2f} ms")
    print(f"p99 Latency:  {p99:.2f} ms")
    
async def main():
    print("Initializing Pipeline...")
    pipeline = SignDetectorPipeline()
    dummy_b64 = create_dummy_image()
    
    # Warmup
    print("Warming up...")
    await pipeline.detect_sign(dummy_b64)
    
    # Baseline (Sequential)
    await run_stress_test(pipeline, dummy_b64, num_requests=20, max_workers=1, force_fast_path=True)
    
    # Concurrent (Moderate Load)
    await run_stress_test(pipeline, dummy_b64, num_requests=50, max_workers=10, force_fast_path=True)
    
    # High Load
    await run_stress_test(pipeline, dummy_b64, num_requests=200, max_workers=50, force_fast_path=True)

if __name__ == "__main__":
    asyncio.run(main())
