import re

try:
    from modules.stage3_retrieval import RetrievalEngine
    from modules.stage4_reliability import ReliabilityAnalyzer
except ImportError:
    from stage3_retrieval import RetrievalEngine
    from stage4_reliability import ReliabilityAnalyzer


class HybridRankingQAEngine:
    def __init__(self, documents: list, link_graph: dict):
        self.documents = documents
        self.retrieval_engine = RetrievalEngine(documents)
        self.reliability_analyzer = ReliabilityAnalyzer(documents, link_graph)

    def execute_search_pipeline(
        self,
        query: str,
        user_interests: list = None,
        alpha: float = 0.55,
        use_query_expansion: bool = True
    ) -> dict:
        retrieval_results, exp_meta = self.retrieval_engine.retrieve_all_models(
            query=query,
            user_interests=user_interests,
            use_query_expansion=use_query_expansion
        )
        retrieval_map = {r["doc_id"]: r for r in retrieval_results}

        reliability_reports, clusters = self.reliability_analyzer.evaluate_all_documents()
        reliability_map = {r["doc_id"]: r for r in reliability_reports}

        hybrid_ranked_docs = []
        for doc in self.documents:
            doc_id = doc["id"]
            rel_data = retrieval_map.get(doc_id, {"combined_relevance": 0.0, "bm25": 0.0})
            cred_data = reliability_map.get(doc_id, {"total_reliability": 0.0})

            ir_score = float(rel_data["combined_relevance"])
            cred_score = float(cred_data["total_reliability"])

            if ir_score > 0.01:
                hybrid_score = (alpha * ir_score) + ((1.0 - alpha) * cred_score)
            else:
                hybrid_score = (0.2 * ir_score) + (0.8 * cred_score)

            hybrid_ranked_docs.append({
                "doc": doc,
                "relevance_score": round(ir_score, 4),
                "reliability_score": round(cred_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "retrieval_details": rel_data,
                "reliability_details": cred_data
            })

        # Sort primarily by Reliability Score (Descending), breaking ties with Hybrid Score
        hybrid_ranked_docs.sort(
            key=lambda x: (x["reliability_score"], x["hybrid_score"]),
            reverse=True
        )

        standard_ranked_docs = sorted(
            hybrid_ranked_docs,
            key=lambda x: x["retrieval_details"]["bm25"],
            reverse=True
        )

        # Extract Answer ONLY from the #1 Most Reliable Source
        top_answer = self._extract_answer_from_most_reliable(hybrid_ranked_docs, query)

        standard_metrics = self._evaluate_metrics(standard_ranked_docs, k=3)
        hybrid_metrics = self._evaluate_metrics(hybrid_ranked_docs, k=3)

        return {
            "query_metadata": exp_meta,
            "hybrid_ranking": hybrid_ranked_docs,
            "standard_ranking": standard_ranked_docs,
            "extracted_answer": top_answer,
            "evaluation": {
                "standard_bm25": standard_metrics,
                "credence_hybrid": hybrid_metrics
            },
            "clusters": clusters
        }

    @staticmethod
    def _extract_answer_from_most_reliable(ranked_docs: list, query: str) -> dict:
        """
        Extracts the best-matching sentence STRICTLY from the #1 most reliable
        (highest trust score) document that matches the query.
        """
        if not ranked_docs:
            return {"answer": "No reliable documents found.", "source": "None"}

        # Filter to the highest reliability tier and pick #1
        most_reliable_item = max(
            ranked_docs,
            key=lambda x: (x["reliability_score"], x["relevance_score"])
        )
        doc = most_reliable_item["doc"]
        full_text = f"{doc.get('title', '')}. {doc.get('text', '')}"
        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        best_sentence = doc.get("text", full_text)
        max_overlap = -1

        for sent in sentences:
            if len(sent.strip()) < 20:
                continue
            sent_terms = set(re.findall(r'\b[a-z]{3,}\b', sent.lower()))
            overlap = len(query_terms.intersection(sent_terms))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sent

        return {
            "answer": best_sentence.strip(),
            "source_doc": doc["id"],
            "source_title": doc.get("title", "Verified Document"),
            "source_domain": doc.get("source", "unknown"),
            "url": doc.get("url", "#"),
            "reliability_score": most_reliable_item["reliability_score"],
            "confidence": most_reliable_item["hybrid_score"]
        }

    @staticmethod
    def _evaluate_metrics(ranked_docs: list, k: int = 3) -> dict:
        top_k = ranked_docs[:k]
        rel_retrieved = sum(1 for item in top_k if item["doc"].get("is_relevant", 0) == 1)
        total_relevant_in_corpus = sum(1 for item in ranked_docs if item["doc"].get("is_relevant", 0) == 1)

        precision = rel_retrieved / k if k > 0 else 0.0
        recall = rel_retrieved / total_relevant_in_corpus if total_relevant_in_corpus > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        hits = 0
        sum_prec = 0.0
        for rank, item in enumerate(ranked_docs, start=1):
            if item["doc"].get("is_relevant", 0) == 1:
                hits += 1
                sum_prec += (hits / rank)
        ap = sum_prec / total_relevant_in_corpus if total_relevant_in_corpus > 0 else 0.0

        return {
            f"Precision@{k}": round(precision, 3),
            f"Recall@{k}": round(recall, 3),
            f"F1-Score@{k}": round(f1, 3),
            "Average Precision (AP)": round(ap, 3)
        }