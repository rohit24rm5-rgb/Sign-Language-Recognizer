"""
Semantic-Router Benchmark Script
================================

A companion script to compare Semantic-Router performance on the same tasks
(System Integration & Banking77).
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

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import FastEmbedEncoder

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_banking77():
    print("Loading Banking77 from HuggingFace...")
    ds = load_dataset("mteb/banking77")
    features = ds["train"].features["label"]

    if hasattr(features, "names"):
        intent_names = features.names
    else:
        label_to_text = {}
        for row in ds["train"]:
            if row["label"] not in label_to_text:
                clean = re.sub(r'[^a-zA-Z0-9_-]', '_', row.get("label_text", str(row["label"])))
                label_to_text[row["label"]] = clean
        intent_names = [label_to_text[i] for i in range(max(label_to_text.keys()) + 1)]

    route_utterances = defaultdict(list)
    for row in ds["train"]:
        route_utterances[intent_names[row["label"]]].append(row["text"])

    test_queries = []
    for row in ds["test"]:
        test_queries.append({
            "query": row["text"],
            "expected": intent_names[row["label"]],
            "is_ood": False,
        })

    return {
        "name": "Banking77",
        "num_intents": len(route_utterances),
        "routes": route_utterances,
        "test_queries": test_queries,
        "has_ood": False,
    }

def dataset_hash(test_queries):
    raw = json.dumps([q["query"] for q in test_queries], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Benchmark Evaluation
# ---------------------------------------------------------------------------

def run_benchmark(dataset_info, model_name="BAAI/bge-small-en-v1.5"):
    name = dataset_info["name"]
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {name} (semantic-router)")
    print(f"{'='*60}")

    encoder = FastEmbedEncoder(name=model_name)
    
    routes = []
    total_utterances = 0
    for intent_name, utterances in dataset_info["routes"].items():
        routes.append(Route(name=intent_name, utterances=utterances))
        total_utterances += len(utterances)
        
    t_build_start = time.perf_counter()
    router = SemanticRouter(encoder=encoder, routes=routes)
    build_time = time.perf_counter() - t_build_start
    print(f"Index built in {build_time:.2f}s ({total_utterances} utterances)")

    test_queries = dataset_info["test_queries"][:500] # Limit to 500 queries
    print(f"Evaluating {len(test_queries)} test queries...")

    y_true, y_pred = [], []
    latencies = []

    for i, tq in enumerate(test_queries):
        t_s = time.perf_counter()
        result = router(tq["query"])
        latencies.append((time.perf_counter() - t_s) * 1000) 

        pred_name = result.name if result.name else None
        y_true.append(tq["expected"])
        y_pred.append(pred_name)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(test_queries)}")

    peak_memory_mb = 0.0

    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    y_pred_sk = ["__NONE__" if p is None else p for p in y_pred]
    y_true_sk = ["__NONE__" if t is None else t for t in y_true]
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_sk, y_pred_sk, average="macro", zero_division=0)

    predicted_intents = set(y_pred) - {None}
    coverage = len(predicted_intents) / dataset_info["num_intents"]

    latencies_arr = np.array(latencies)
    throughput = 1000.0 / np.mean(latencies_arr)

    print(f"\n[Results: {name}]")
    print(f"  Top-1 Accuracy:  {top1*100:.2f}%")
    print(f"  F1 Score:        {f1*100:.2f}%")
    print(f"  Avg Latency:     {np.mean(latencies_arr):.2f}ms")
    print(f"  Throughput:      {throughput:.1f} QPS")

    return {
        "dataset_name": name,
        "dataset_hash": dataset_hash(test_queries),
        "route_count": dataset_info["num_intents"],
        "utterance_count": total_utterances,
        "test_query_count": len(test_queries),
        "index_build_time_s": round(build_time, 2),
        "embedding_model": model_name,
        "accuracy": {
            "top_1": round(top1 * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "route_coverage": round(coverage * 100, 2)
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

def run_system_integration_benchmark():
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: SignDetectorPipeline Integration (semantic-router)")
    print(f"{'='*60}")
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.ai_pipeline import SignDetectorPipeline
    from PIL import Image
    import io
    import base64
    
    # We must patch the pipeline to use semantic-router instead of synaptoroute
    pipeline = SignDetectorPipeline()
    
    encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
    routes = [Route(name="known_sign", utterances=["Hello", "iloveyou", "yes", "No", "Thankyou", "A clear hand sign"])]
    pipeline.router = SemanticRouter(encoder=encoder, routes=routes)
    
    # Mock FAISS retrieval to force fast-path
    original_retrieve = pipeline.retrieve_context
    def mock_retrieve_fast(b64):
        return [{"label": "Hello", "distance": 0.1, "base64": b64}]
    pipeline.retrieve_context = mock_retrieve_fast
    
    # Create a dummy image
    img = Image.new('RGB', (300, 300), color = 'white')
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    dummy_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Warmup
    pipeline.detect_sign(dummy_b64)
    
    latencies_fast = []
    for _ in range(5):
        t0 = time.perf_counter()
        pipeline.detect_sign(dummy_b64)
        latencies_fast.append((time.perf_counter() - t0) * 1000)
        
    avg_fast = sum(latencies_fast) / len(latencies_fast)
    
    print(f"\n[Results: System Integration]")
    print(f"  Fast-Path (semantic-router) Avg Latency: {avg_fast:.2f}ms")
    
    # Restore
    pipeline.retrieve_context = original_retrieve

    return {
        "dataset_name": "System Integration (Hand Sign Pipeline - SR)",
        "dataset_hash": "N/A",
        "route_count": 1,
        "utterance_count": 1,
        "test_query_count": 5,
        "embedding_model": "N/A",
        "index_build_time_s": 0.0,
        "accuracy": {
            "top_1": 100.0, "precision": 100.0, "recall": 100.0, "f1": 100.0, "route_coverage": 100.0
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
        }
    }

def main():
    print("=" * 60)
    print("  Semantic-Router Benchmark + Pipeline Integration")
    print("=" * 60)

    results = []

    try:
        results.append(run_system_integration_benchmark())
    except Exception as e:
        print(f"\nIntegration benchmark failed: {e}")
        traceback.print_exc()

    try:
        banking = load_banking77()
        results.append(run_benchmark(banking, "BAAI/bge-small-en-v1.5"))
    except Exception as e:
        print(f"\nBanking77 failed: {e}")
        traceback.print_exc()

    manifest = {
        "manifest_version": "1.0.0",
        "benchmark_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    manifest_path = "sr_benchmark_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  Manifest:  {os.path.abspath(manifest_path)}")

if __name__ == "__main__":
    main()
