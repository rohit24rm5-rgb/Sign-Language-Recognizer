# SynaptoRoute Benchmark & System Integration Report

**Benchmark ID:** `b578e7f3-7ff5-4a38-b52a-29ded6b62a65`
**Timestamp:** 2026-06-02T05:53:21.580206+00:00
**SynaptoRoute Version:** 0.3.1

---

## System Configuration

| Field | Value |
|---|---|
| OS | Windows 10.0.26200 |
| CPU | Intel64 Family 6 Model 186 Stepping 3, GenuineIntel |
| Cores / Threads | 10 / 12 |
| RAM | 15.69 GB (available: 3.98 GB) |
| GPU | none (0 MB) |
| Python | 3.11.0 |
| FastEmbed | 0.8.0 |
| ONNX Runtime | 1.26.0 |
| NumPy | 2.4.6 |
| Faiss | 1.14.2 |

---

## System Integration (Hand Sign Pipeline)

- **Routes:** 1
- **Utterances:** 1
- **Test Queries:** 8
- **Model:** `N/A`
- **Index Build Time:** 0.0s

### Accuracy

| Metric | Value |
|---|---|
| Top-1 Accuracy | 100.0% |
| Precision | 100.0% |
| Recall | 100.0% |
| F1 Score | 100.0% |
| Route Coverage | 100.0% |

### Performance

| Metric | Value |
|---|---|
| Avg Latency | 18.16ms |
| Throughput | 55.1 QPS |
| Peak Memory | 0.0 MB |
| **LLM Fallback Latency** | **646.9ms** |
| **Integration Speedup** | **35.63x** |

---

## Banking77

- **Routes:** 4
- **Utterances:** 16
- **Test Queries:** 10
- **Model:** `BAAI/bge-small-en-v1.5`
- **Index Build Time:** 0.17s

### Accuracy

| Metric | Value |
|---|---|
| Top-1 Accuracy | 70.0% |
| Precision | 70.0% |
| Recall | 70.0% |
| F1 Score | 66.67% |
| Route Coverage | 100.0% |
| OOD Rejection | 0.0% |

### Performance

| Metric | Value |
|---|---|
| Avg Latency | 16.93ms |
| Throughput | 59.1 QPS |
| Peak Memory | 0.0 MB |
