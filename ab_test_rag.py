"""RAG AB Test: bge-small-zh-v1.5 vs bge-m3, with/without reranker.
Usage: python ab_test_rag.py
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_kb.json")

DEVICE = os.environ.get("RAG_DEVICE", "cuda:0")

BGE_SMALL_DIR = os.path.join(MODEL_DIR, "bge-small-zh-v1.5")
BGE_M3_DIR = os.path.join(MODEL_DIR, "bge-m3")
BGE_M3_MS_DIR = os.path.join(MODEL_DIR, "BAAI", "bge-m3")
RERANKER_DIR = os.path.join(MODEL_DIR, "bge-reranker-v2-m3")

TEST_QUERIES = [
    "你们的价格是多少",
    "怎么联系客服",
    "支持马来语吗",
    "AI可以处理什么问题",
    "数据安全怎么保障",
    "能同时接多少电话",
    "声音可以自定义吗",
    "怎么开始试用",
    "退款怎么办",
    "你们在哪个城市",
]


def load_docs():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_config(name, embed_dir, reranker_dir, docs, queries, device):
    from engine.rag import RAGEngine

    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    rag = RAGEngine(
        embed_model_dir=embed_dir,
        device=device,
        reranker_model_dir=reranker_dir,
        top_k=3,
        rerank_top_k=5,
    ).load()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model load: {load_ms:.0f}ms")

    build = rag.build_index(docs)
    print(f"  Index build: {build['encode_ms']:.0f}ms ({build['num_docs']} docs, dim={build['dim']})")
    print(f"  Per-doc encode: {build['encode_per_doc_ms']:.1f}ms")

    # Warmup
    rag.query("测试查询")

    latencies = {"embed": [], "search": [], "rerank": [], "total": []}
    results_log = []

    for q in queries:
        result = rag.query(q)
        latencies["embed"].append(result["embed_ms"])
        latencies["search"].append(result["search_ms"])
        latencies["rerank"].append(result["rerank_ms"])
        latencies["total"].append(result["total_ms"])
        top1 = result["results"][0] if result["results"] else {}
        results_log.append({
            "query": q,
            "top1_q": top1.get("question", ""),
            "score": top1.get("score", 0),
        })

    print(f"\n  Latency ({len(queries)} queries):")
    for k, v in latencies.items():
        arr = np.array(v)
        print(f"    {k:>8s}: avg={arr.mean():.1f}ms  p50={np.median(arr):.1f}ms  "
              f"min={arr.min():.1f}ms  max={arr.max():.1f}ms")

    print(f"\n  Retrieval quality (top-1):")
    for r in results_log[:5]:
        print(f"    Q: {r['query']}")
        print(f"    → {r['top1_q']} (score={r['score']:.3f})")

    return {
        "name": name,
        "load_ms": load_ms,
        "avg_total_ms": np.mean(latencies["total"]),
        "avg_embed_ms": np.mean(latencies["embed"]),
        "avg_rerank_ms": np.mean(latencies["rerank"]),
    }


def main():
    docs = load_docs()
    print(f"Loaded {len(docs)} documents")
    print(f"Device: {DEVICE}")

    results = []

    if os.path.isdir(BGE_SMALL_DIR):
        r = benchmark_config(
            "bge-small-zh-v1.5 (no reranker)",
            BGE_SMALL_DIR, None, docs, TEST_QUERIES, DEVICE
        )
        results.append(r)

        if os.path.isdir(RERANKER_DIR):
            r = benchmark_config(
                "bge-small-zh-v1.5 + reranker-v2-m3",
                BGE_SMALL_DIR, RERANKER_DIR, docs, TEST_QUERIES, DEVICE
            )
            results.append(r)
    else:
        print(f"SKIP: {BGE_SMALL_DIR} not found")

    m3_dir = BGE_M3_DIR if os.path.isdir(BGE_M3_DIR) else BGE_M3_MS_DIR
    if os.path.isdir(m3_dir):
        r = benchmark_config(
            "bge-m3 (no reranker)",
            m3_dir, None, docs, TEST_QUERIES, DEVICE
        )
        results.append(r)

        if os.path.isdir(RERANKER_DIR):
            r = benchmark_config(
                "bge-m3 + reranker-v2-m3",
                m3_dir, RERANKER_DIR, docs, TEST_QUERIES, DEVICE
            )
            results.append(r)
    else:
        print(f"SKIP: bge-m3 not found at {BGE_M3_DIR} or {BGE_M3_MS_DIR}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<40s} {'Load':>8s} {'Embed':>8s} {'Rerank':>8s} {'Total':>8s}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<40s} {r['load_ms']:>7.0f}ms {r['avg_embed_ms']:>6.1f}ms "
              f"{r['avg_rerank_ms']:>6.1f}ms {r['avg_total_ms']:>6.1f}ms")


if __name__ == "__main__":
    main()
