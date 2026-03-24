"""RAG engine: embedding + FAISS + optional reranker.

Supports two embedding backends:
  - bge-small-zh-v1.5 (24M, ~3-5ms/query, 512-dim) — fastest
  - bge-m3 (568M, ~15-20ms/query, 1024-dim) — best quality

Optional reranker: bge-reranker-v2-m3 (568M, ~10-15ms/top-5)
"""
import os
import json
import time
import numpy as np
import faiss


class RAGEngine:
    def __init__(self, embed_model_dir: str, device: str = "cuda:0",
                 reranker_model_dir: str = None, top_k: int = 3,
                 rerank_top_k: int = 5, score_threshold: float = 0.0):
        self._embed_model_dir = embed_model_dir
        self._reranker_model_dir = reranker_model_dir
        self._device = device
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self._score_threshold = score_threshold

        self._embedder = None
        self._reranker = None
        self._index = None
        self._documents = []
        self._doc_embeddings = None

    def load(self):
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(
            self._embed_model_dir,
            device=self._device,
        )

        if self._reranker_model_dir and os.path.isdir(self._reranker_model_dir):
            from FlagEmbedding import FlagReranker
            self._reranker = FlagReranker(
                self._reranker_model_dir,
                use_fp16=True,
                device=self._device,
            )

        return self

    def build_index(self, documents: list[dict]):
        """Build FAISS index from documents.
        Each doc: {'id': str, 'question': str, 'answer': str, 'text': str (optional)}
        If 'text' is not provided, 'question' + ' ' + 'answer' is used for embedding.
        """
        self._documents = documents
        texts = []
        for doc in documents:
            text = doc.get("text", f"{doc.get('question', '')} {doc.get('answer', '')}")
            texts.append(text.strip())

        t0 = time.perf_counter()
        embeddings = self._embedder.encode(
            texts, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,
        )
        encode_ms = (time.perf_counter() - t0) * 1000

        self._doc_embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._doc_embeddings)

        return {
            "num_docs": len(documents),
            "dim": dim,
            "encode_ms": encode_ms,
            "encode_per_doc_ms": encode_ms / len(documents),
        }

    def query(self, question: str, top_k: int = None) -> dict:
        """Query the RAG index. Returns top-k results with latency info."""
        k = top_k or self._top_k
        rerank_k = max(k, self._rerank_top_k) if self._reranker else k

        t0 = time.perf_counter()
        q_emb = self._embedder.encode(
            [question], normalize_embeddings=True,
        ).astype(np.float32)
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        scores, indices = self._index.search(q_emb, rerank_k)
        search_ms = (time.perf_counter() - t1) * 1000

        candidates = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            doc = self._documents[idx].copy()
            doc["score"] = float(score)
            doc["rank"] = i
            candidates.append(doc)

        rerank_ms = 0.0
        if self._reranker and candidates:
            t2 = time.perf_counter()
            pairs = [[question, c.get("answer", c.get("text", ""))] for c in candidates]
            rerank_scores = self._reranker.compute_score(pairs)
            if isinstance(rerank_scores, (int, float)):
                rerank_scores = [rerank_scores]
            rerank_ms = (time.perf_counter() - t2) * 1000

            for c, rs in zip(candidates, rerank_scores):
                c["rerank_score"] = float(rs)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        results = candidates[:k]
        if self._score_threshold > 0:
            score_key = "rerank_score" if self._reranker else "score"
            results = [r for r in results if r.get(score_key, 0) >= self._score_threshold]
        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "results": results,
            "embed_ms": embed_ms,
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "total_ms": total_ms,
        }

    def get_context(self, question: str, top_k: int = None, max_chars: int = 800) -> dict:
        """Get formatted context string for LLM prompt injection."""
        result = self.query(question, top_k)
        context_parts = []
        for r in result["results"]:
            answer = r.get("answer", r.get("text", ""))
            q = r.get("question", "")
            if q:
                context_parts.append(f"Q: {q}\nA: {answer}")
            else:
                context_parts.append(answer)

        context = "\n\n".join(context_parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "..."

        return {
            "context": context,
            "num_results": len(result["results"]),
            "embed_ms": result["embed_ms"],
            "search_ms": result["search_ms"],
            "rerank_ms": result["rerank_ms"],
            "total_ms": result["total_ms"],
        }

    def save_index(self, path: str):
        """Save FAISS index and documents to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self._index, path + ".faiss")
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

    def load_index(self, path: str):
        """Load FAISS index and documents from disk."""
        self._index = faiss.read_index(path + ".faiss")
        with open(path + ".json", "r", encoding="utf-8") as f:
            self._documents = json.load(f)
        return self
