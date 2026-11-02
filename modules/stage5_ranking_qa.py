import re
import numpy as np

try:
    from modules.stage3_retrieval import RetrievalEngine
    from modules.stage4_reliability import ReliabilityAnalyzer
except ImportError:
    from stage3_retrieval import RetrievalEngine
    from stage4_reliability import ReliabilityAnalyzer


# =====================================================================
# STAGE 5: HYBRID RANKING, QUESTION ANSWERING & IR EVALUATION
# Syllabus Module 3/5 (Document Ranking & QA Systems)
# Syllabus Module 4 (Evaluation: Precision, Recall, F1, Average Precision)
# =====================================================================
class HybridRankingQAEngine:
    """
    Combines IR relevance, reliability mining, hybrid ranking score calculation,
    extractive question answering, and formal IR evaluation metrics.
    """

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
        """
        Executes the complete pipeline:
        Query -> PRF Expansion -> BM25/VSM/Semantic -> PageRank/NB Reliability -> Hybrid Ranking -> QA -> Metrics.
        """
        # 1. Stage 3: Retrieval & Query Expansion
        retrieval_results, exp_meta = self.retrieval_engine.retrieve_all_models(
            query=query,
            user_interests=user_interests,
            use_query_expansion=use_query_expansion
        )
        retrieval_map = {r["doc_id"]: r for r in retrieval_results}

        # 2. Stage 4: Reliability & Authority Mining
        reliability_reports, clusters = self.reliability_analyzer.evaluate_all_documents()
        reliability_map = {r["doc_id"]: r for r in reliability_reports}

        # 3. Stage 5: Hybrid Ranking Formula
        # Final Score = alpha * (IR Relevance) + (1 - alpha) * (Reliability Score)
        hybrid_ranked_docs = []
        for doc in self.documents:
            doc_id = doc["id"]
            rel_data = retrieval_map.get(doc_id, {"combined_relevance": 0.0, "bm25": 0.0})
            cred_data = reliability_map.get(doc_id, {"total_reliability": 0.0})

            ir_score = float(rel_data["combined_relevance"])
            cred_score = float(cred_data["total_reliability"])

            # Only rank documents that have some baseline relevance
            if ir_score > 0.01:
                hybrid_score = (alpha * ir_score) + ((1.0 - alpha) * cred_score)
            else:
                hybrid_score = 0.0

            hybrid_ranked_docs.append({
                "doc": doc,
                "relevance_score": round(ir_score, 4),
                "reliability_score": round(cred_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "retrieval_details": rel_data,
                "reliability_details": cred_data
            })

        # Sort descending by hybrid score
        hybrid_ranked_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Also sort by BM25 only (for baseline comparison)
        standard_ranked_docs = sorted(
            hybrid_ranked_docs,
            key=lambda x: x["retrieval_details"]["bm25"],
            reverse=True
        )

        # 4. Module 5: Extractive Question Answering (QA) on top result
        top_answer = self._extract_answer(hybrid_ranked_docs, query)

        # 5. Module 4: Performance Evaluation Metrics (Precision, Recall, F1, AP)
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
    def _extract_answer(ranked_docs: list, query: str) -> dict:
        """Module 5: Question Answering System (Extracts sentence answering query from top document)."""
        if not ranked_docs:
            return {"answer": "No relevant documents found.", "source": "None"}

        top_item = ranked_docs[0]
        doc = top_item["doc"]
        full_text = f"{doc.get('title', '')}. {doc.get('text', '')}"
        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        best_sentence = sentences[0] if sentences else full_text
        max_overlap = -1

        for sent in sentences:
            sent_terms = set(re.findall(r'\b[a-z]{3,}\b', sent.lower()))
            overlap = len(query_terms.intersection(sent_terms))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sent

        return {
            "answer": best_sentence.strip(),
            "source_doc": doc["id"],
            "source_domain": doc.get("source", "unknown"),
            "url": doc.get("url", "#"),
            "confidence": top_item["hybrid_score"]
        }

    @staticmethod
    def _evaluate_metrics(ranked_docs: list, k: int = 3) -> dict:
        """Module 4: Evaluation of IR Systems (Precision@K, Recall@K, F1, Average Precision)."""
        top_k = ranked_docs[:k]
        rel_retrieved = sum(1 for item in top_k if item["doc"].get("is_relevant", 0) == 1)
        total_relevant_in_corpus = sum(1 for item in ranked_docs if item["doc"].get("is_relevant", 0) == 1)

        precision = rel_retrieved / k if k > 0 else 0.0
        recall = rel_retrieved / total_relevant_in_corpus if total_relevant_in_corpus > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Average Precision (AP)
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


# =====================================================================
# QUICK TEST FOR STAGE 5 (CLI RUNNER)
# =====================================================================
if __name__ == "__main__":
    try:
        from modules.stage1_ingestion_crawler import WebAndSocialCrawler
    except ImportError:
        from stage1_ingestion_crawler import WebAndSocialCrawler

    print("=== STAGE 5: HYBRID RANKING, QA & EVALUATION METRICS TEST ===")
    crawler = WebAndSocialCrawler()
    docs, link_graph = crawler.crawl_and_collect("merit scholarship admission")

    engine = HybridRankingQAEngine(docs, link_graph)
    sample_query = "merit scholarship university admission"
    user_profile = ["undergraduate", "official", "financial"]

    output = engine.execute_search_pipeline(
        query=sample_query,
        user_interests=user_profile,
        alpha=0.55,
        use_query_expansion=True
    )

    print(f"\n[Query Expansion]: Original: '{output['query_metadata']['original_query']}'")
    print(f"                 Expanded: '{output['query_metadata']['expanded_query']}'")
    print(f"\n[Question Answering System]:")
    print(f"  Answer : {output['extracted_answer']['answer']}")
    print(f"  Source : {output['extracted_answer']['source_domain']} ({output['extracted_answer']['source_doc']})")

    print(f"\n[Evaluation Comparison (Top K=3)]:")
    print(f"  Standard BM25 Only   : {output['evaluation']['standard_bm25']}")
    print(f"  Credence Hybrid IR   : {output['evaluation']['credence_hybrid']}")

    print(f"\n--- CREDENCE HYBRID RANKED RESULTS TABLE ---")
    print(f"{'Rank':<4} | {'ID':<6} | {'Hybrid':<7} | {'Relev.':<7} | {'Reliab.':<8} | {'Source':<20} | {'Title'}")
    print("-" * 92)
    for r, item in enumerate(output["hybrid_ranking"], 1):
        doc = item["doc"]
        print(f"{r:<4} | {doc['id']:<6} | {item['hybrid_score']:<7} | "
              f"{item['relevance_score']:<7} | {item['reliability_score']:<8} | "
              f"{doc['source']:<20} | {doc['title'][:32]}")