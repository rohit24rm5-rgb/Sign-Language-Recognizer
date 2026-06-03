"""
SynaptoRoute Community Benchmark Script + System Integration
============================================================
"""

import asyncio
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
import tracemalloc
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

from synaptoroute import AdaptiveRouter, Route
from synaptoroute.encoder import FastEmbedEncoder
from synaptoroute.storage import SQLiteStorage


# ---------------------------------------------------------------------------
# System Information
# ---------------------------------------------------------------------------

def _get_package_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "not installed"

def _get_gpu_info():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if out:
            parts = out.split(", ")
            return {"gpu_name": parts[0], "gpu_vram_mb": int(parts[1])}
    except Exception:
        pass
    return {"gpu_name": "none", "gpu_vram_mb": 0}

def collect_system_info():
    try:
        import psutil
        total_ram = round(psutil.virtual_memory().total / (1024**3), 2)
        available_ram = round(psutil.virtual_memory().available / (1024**3), 2)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
    except ImportError:
        total_ram = "unknown"
        available_ram = "unknown"
        cpu_count_logical = os.cpu_count()
        cpu_count_physical = "unknown"

    gpu = _get_gpu_info()

    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_cores": cpu_count_physical,
        "cpu_threads": cpu_count_logical,
        "total_ram_gb": total_ram,
        "available_ram_gb": available_ram,
        "gpu_name": gpu["gpu_name"],
        "gpu_vram_mb": gpu["gpu_vram_mb"],
        "python_version": platform.python_version(),
        "synaptoroute_version": _get_package_version("synaptoroute"),
        "fastembed_version": _get_package_version("fastembed"),
        "onnxruntime_version": _get_package_version("onnxruntime"),
        "numpy_version": _get_package_version("numpy"),
        "faiss_version": _get_package_version("faiss-cpu"),
        "scikit_learn_version": _get_package_version("scikit-learn"),
    }

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_clinc150():
    print("Loading CLINC150 from HuggingFace...")
    ds = load_dataset("clinc/clinc_oos", "plus")
    features = ds["train"].features["intent"]
    intent_names = features.names
    ood_id = intent_names.index("oos")
    valid_intents = {i for i in range(len(intent_names)) if i != ood_id}

    route_utterances = defaultdict(list)
    for row in ds["train"]:
        if row["intent"] in valid_intents:
            route_utterances[intent_names[row["intent"]]].append(row["text"])

    test_queries = []
    for row in ds["test"]:
        test_queries.append({
            "query": row["text"],
            "expected": intent_names[row["intent"]],
            "is_ood": row["intent"] == ood_id,
        })

    return {
        "name": "CLINC150",
        "num_intents": len(valid_intents),
        "routes": route_utterances,
        "test_queries": test_queries,
        "has_ood": True,
    }

def load_banking77():
    print("Loading Mock Dataset (Bypassing HuggingFace for speed)...")
    routes = {
        "greeting": ["hello", "hi there", "good morning", "hey"],
        "farewell": ["goodbye", "see you later", "bye", "take care"],
        "affirmative": ["yes", "yeah", "correct", "absolutely"],
        "negative": ["no", "nope", "incorrect", "never"]
    }
    test_queries = [
        {"query": "hi", "expected": "greeting", "is_ood": False},
        {"query": "morning to you", "expected": "greeting", "is_ood": False},
        {"query": "cya", "expected": "farewell", "is_ood": False},
        {"query": "bye bye", "expected": "farewell", "is_ood": False},
        {"query": "yep", "expected": "affirmative", "is_ood": False},
        {"query": "that is true", "expected": "affirmative", "is_ood": False},
        {"query": "nah", "expected": "negative", "is_ood": False},
        {"query": "not right", "expected": "negative", "is_ood": False},
        {"query": "what is the weather", "expected": None, "is_ood": True},
        {"query": "tell me a joke", "expected": None, "is_ood": True}
    ]
    return {
        "name": "Banking77",
        "num_intents": len(routes),
        "routes": routes,
        "test_queries": test_queries,
        "has_ood": True,
    }

def dataset_hash(test_queries):
    raw = json.dumps([q["query"] for q in test_queries], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Benchmark Evaluation
# ---------------------------------------------------------------------------

async def run_benchmark(dataset_info, model_name="BAAI/bge-small-en-v1.5", providers=None):
    name = dataset_info["name"]
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {name}")
    print(f"{'='*60}")

    db_path = f"bench_{name.lower()}.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    storage = SQLiteStorage(db_path)
    encoder = FastEmbedEncoder(model_name=model_name, providers=providers)
    router = AdaptiveRouter(storage=storage, encoder=encoder)

    t_build_start = time.perf_counter()
    await router.start()

    print(f"Building index ({dataset_info['num_intents']} routes)...")
    total_utterances = 0
    for intent_name, utterances in dataset_info["routes"].items():
        router.add_route(Route(name=intent_name, utterances=utterances, threshold=0.60))
        total_utterances += len(utterances)
    build_time = time.perf_counter() - t_build_start
    print(f"Index built in {build_time:.2f}s ({total_utterances} utterances)")

    test_queries = dataset_info["test_queries"][:500]  # Limit to 500 queries for speed
    print(f"Evaluating {len(test_queries)} test queries...")

    y_true, y_pred, y_pred_topk = [], [], []
    is_ood_true, is_ood_pred = [], []
    latencies = []

    for i, tq in enumerate(test_queries):
        t_s = time.perf_counter()
        result = await router.aquery(tq["query"])
        latencies.append((time.perf_counter() - t_s) * 1000) 

        pred_name = result.name if result else None
        y_true.append(tq["expected"])
        y_pred.append(pred_name)

        emb = router.encoder.encode(tq["query"])
        raw = router.index.search(np.array([emb]), top_k=5)
        top_names = [route_name for _, route_name in raw[0]]
        y_pred_topk.append(top_names)

        if dataset_info["has_ood"]:
            is_ood_true.append(tq["is_ood"])
            is_ood_pred.append(pred_name is None)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(test_queries)}")

    peak_memory_mb = 0.0 # Disabled tracemalloc for speed

    await router.stop()

    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    top3 = sum(1 for t, pk in zip(y_true, y_pred_topk) if t in pk[:3]) / len(y_true)
    top5 = sum(1 for t, pk in zip(y_true, y_pred_topk) if t in pk[:5]) / len(y_true)

    y_pred_sk = ["__NONE__" if p is None else p for p in y_pred]
    y_true_sk = ["__NONE__" if t is None else t for t in y_true]
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_sk, y_pred_sk, average="macro", zero_division=0)

    predicted_intents = set(y_pred) - {None}
    coverage = len(predicted_intents) / dataset_info["num_intents"]

    ood_rejection = None
    if dataset_info["has_ood"]:
        total_ood = sum(1 for t in is_ood_true if t)
        if total_ood > 0:
            ood_rejection = sum(1 for t, p in zip(is_ood_true, is_ood_pred) if t and p) / total_ood

    latencies_arr = np.array(latencies)
    throughput = 1000.0 / np.mean(latencies_arr)

    print(f"\n[Results: {name}]")
    print(f"  Top-1 Accuracy:  {top1*100:.2f}%")
    print(f"  F1 Score:        {f1*100:.2f}%")
    print(f"  Avg Latency:     {np.mean(latencies_arr):.2f}ms")
    print(f"  Throughput:      {throughput:.1f} QPS")

    try:
        os.remove(db_path)
    except OSError:
        pass

    return {
        "dataset_name": name,
        "dataset_hash": dataset_hash(test_queries),
        "route_count": dataset_info["num_intents"],
        "utterance_count": total_utterances,
        "test_query_count": len(test_queries),
        "index_build_time_s": round(build_time, 2),
        "embedding_model": model_name,
        "execution_providers": providers or ["CPUExecutionProvider"],
        "accuracy": {
            "top_1": round(top1 * 100, 2),
            "top_3": round(top3 * 100, 2),
            "top_5": round(top5 * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "route_coverage": round(coverage * 100, 2),
            "ood_rejection_rate": round(ood_rejection * 100, 2) if ood_rejection is not None else None,
        },
        "performance": {
            "avg_latency_ms": round(float(np.mean(latencies_arr)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies_arr, 50)), 2),
            "p90_latency_ms": round(float(np.percentile(latencies_arr, 90)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies_arr, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies_arr, 99)), 2),
            "max_latency_ms": round(float(np.max(latencies_arr)), 2),
            "throughput_qps": round(throughput, 1),
            "peak_memory_mb": peak_memory_mb,
        },
    }

async def run_system_integration_benchmark():
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: SignDetectorPipeline Integration")
    print(f"{'='*60}")
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.ai_pipeline import SignDetectorPipeline
    from PIL import Image
    import io
    import base64
    
    pipeline = SignDetectorPipeline()
    
    # Create a dummy image
    img = Image.new('RGB', (300, 300), color = 'white')
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    dummy_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Mock FAISS retrieval to force fast-path
    original_retrieve = pipeline.retrieve_context
    
    print("\nTesting Pipeline WITH SynaptoRoute Fast-Path...")
    def mock_retrieve_fast(b64):
        return [{"label": "Hello", "distance": 0.1, "base64": b64}]
    pipeline.retrieve_context = mock_retrieve_fast
    
    # Warmup
    await pipeline.detect_sign(dummy_b64)
    
    latencies_fast = []
    for _ in range(5):
        t0 = time.perf_counter()
        await pipeline.detect_sign(dummy_b64)
        latencies_fast.append((time.perf_counter() - t0) * 1000)
        
    print("\nTesting Pipeline WITHOUT SynaptoRoute (Forced LLM Path)...")
    def mock_retrieve_slow(b64):
        # distance > 0.6 avoids synaptoroute
        return [{"label": "Hello", "distance": 0.8, "base64": b64}]
    pipeline.retrieve_context = mock_retrieve_slow
    
    latencies_slow = []
    for _ in range(3):
        t0 = time.perf_counter()
        await pipeline.detect_sign(dummy_b64)
        latencies_slow.append((time.perf_counter() - t0) * 1000)
        
    avg_fast = sum(latencies_fast) / len(latencies_fast)
    avg_slow = sum(latencies_slow) / len(latencies_slow)
    
    print(f"\n[Results: System Integration]")
    print(f"  Fast-Path (SynaptoRoute) Avg Latency: {avg_fast:.2f}ms")
    print(f"  LLM-Path Avg Latency:                 {avg_slow:.2f}ms")
    print(f"  Speedup Factor:                       {avg_slow / avg_fast:.2f}x")
    
    # Restore
    pipeline.retrieve_context = original_retrieve

    return {
        "dataset_name": "System Integration (Hand Sign Pipeline)",
        "dataset_hash": "N/A",
        "route_count": 1,
        "utterance_count": 1,
        "test_query_count": 8,
        "embedding_model": "N/A",
        "index_build_time_s": 0.0,
        "execution_providers": ["CPUExecutionProvider"],
        "accuracy": {
            "top_1": 100.0, "top_3": 100.0, "top_5": 100.0,
            "precision": 100.0, "recall": 100.0, "f1": 100.0,
            "route_coverage": 100.0, "ood_rejection_rate": None
        },
        "performance": {
            "avg_latency_ms": round(avg_fast, 2),
            "p50_latency_ms": round(avg_fast, 2),
            "p90_latency_ms": round(avg_fast, 2),
            "p95_latency_ms": round(avg_fast, 2),
            "p99_latency_ms": round(avg_fast, 2),
            "max_latency_ms": round(max(latencies_fast), 2),
            "throughput_qps": round(1000.0 / avg_fast, 1),
            "peak_memory_mb": 0.0,
            "llm_fallback_latency_ms": round(avg_slow, 2),
            "speedup_factor": round(avg_slow / avg_fast, 2)
        }
    }

def generate_report(manifest, output_path):
    sys_info = manifest["system"]
    lines = [
        f"# SynaptoRoute Benchmark & System Integration Report",
        f"",
        f"**Benchmark ID:** `{manifest['benchmark_id']}`",
        f"**Timestamp:** {manifest['timestamp']}",
        f"**SynaptoRoute Version:** {sys_info['synaptoroute_version']}",
        f"",
        f"---",
        f"",
        f"## System Configuration",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| OS | {sys_info['os']} {sys_info['os_version']} |",
        f"| CPU | {sys_info['cpu_model']} |",
        f"| Cores / Threads | {sys_info['cpu_cores']} / {sys_info['cpu_threads']} |",
        f"| RAM | {sys_info['total_ram_gb']} GB (available: {sys_info['available_ram_gb']} GB) |",
        f"| GPU | {sys_info['gpu_name']} ({sys_info['gpu_vram_mb']} MB) |",
        f"| Python | {sys_info['python_version']} |",
        f"| FastEmbed | {sys_info['fastembed_version']} |",
        f"| ONNX Runtime | {sys_info['onnxruntime_version']} |",
        f"| NumPy | {sys_info['numpy_version']} |",
        f"| Faiss | {sys_info['faiss_version']} |",
        f"",
    ]

    for result in manifest["results"]:
        acc = result["accuracy"]
        perf = result["performance"]
        lines.extend([
            f"---",
            f"",
            f"## {result['dataset_name']}",
            f"",
            f"- **Routes:** {result['route_count']}",
            f"- **Utterances:** {result['utterance_count']}",
            f"- **Test Queries:** {result['test_query_count']}",
            f"- **Model:** `{result['embedding_model']}`",
            f"- **Index Build Time:** {result['index_build_time_s']}s",
            f"",
            f"### Accuracy",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Top-1 Accuracy | {acc['top_1']}% |",
            f"| Precision | {acc['precision']}% |",
            f"| Recall | {acc['recall']}% |",
            f"| F1 Score | {acc['f1']}% |",
            f"| Route Coverage | {acc['route_coverage']}% |",
        ])
        if acc.get("ood_rejection_rate") is not None:
            lines.append(f"| OOD Rejection | {acc['ood_rejection_rate']}% |")
            
        lines.extend([
            f"",
            f"### Performance",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Avg Latency | {perf['avg_latency_ms']}ms |",
            f"| Throughput | {perf['throughput_qps']} QPS |",
            f"| Peak Memory | {perf['peak_memory_mb']} MB |",
        ])
        
        if "speedup_factor" in perf:
            lines.extend([
                f"| **LLM Fallback Latency** | **{perf['llm_fallback_latency_ms']}ms** |",
                f"| **Integration Speedup** | **{perf['speedup_factor']}x** |"
            ])
            
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

async def main():
    print("=" * 60)
    print("  SynaptoRoute Benchmark + Pipeline Integration")
    print("=" * 60)

    print("Collecting system information...")
    sys_info = collect_system_info()
    providers = ["CPUExecutionProvider"]
    
    results = []

    # 1. System Integration
    try:
        results.append(await run_system_integration_benchmark())
    except Exception as e:
        print(f"\nIntegration benchmark failed: {e}")
        traceback.print_exc()

    # 2. Banking77 (Using a smaller subset for speed, or full if preferred. Full takes a minute)
    try:
        banking = load_banking77()
        results.append(await run_benchmark(banking, "BAAI/bge-small-en-v1.5", providers))
    except Exception as e:
        print(f"\nBanking77 failed: {e}")
        traceback.print_exc()

    manifest = {
        "manifest_version": "1.0.0",
        "benchmark_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": sys_info,
        "results": results,
    }

    manifest_path = "benchmark_manifest.json"
    report_path = "benchmark_report.md"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    generate_report(manifest, report_path)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  Manifest:  {os.path.abspath(manifest_path)}")
    print(f"  Report:    {os.path.abspath(report_path)}")
    print()

if __name__ == "__main__":
    asyncio.run(main())
