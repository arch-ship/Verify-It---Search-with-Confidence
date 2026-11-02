import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

try:
    from modules.stage2_indexer import TextPreprocessor
except ImportError:
    from stage2_indexer import TextPreprocessor


# =====================================================================
# STAGE 3: MULTI-MODEL INFORMATION RETRIEVAL ENGINE
# Syllabus Module 3 (VSM, TF-IDF, BM25, Relevance Feedback, Query Expansion)
# Syllabus Module 5 (Semantic Search & Personalized Search)
# =====================================================================
class RetrievalEngine:
    """
    Implements:
    1. Vector Space Model (TF-IDF + Cosine Similarity)
    2. Probabilistic Retrieval (Okapi BM25)
    3. Semantic Search (Latent Semantic Analysis via TruncatedSVD)
    4. Relevance Feedback & Query Expansion (Pseudo-Relevance Feedback)
    5. Personalized Search (User Profile Re-weighting)
    """

    def __init__(self, documents: list):
        self.documents = documents
        self.N = len(documents)

        # Preprocess corpus using Stage 2 TextPreprocessor
        self.tokenized_corpus = [
            TextPreprocessor.preprocess(f"{doc.get('title', '')} {doc.get('text', '')}")
            for doc in documents
        ]
        self.joined_corpus = [" ".join(tokens) for tokens in self.tokenized_corpus]

        # 1. Build TF-IDF Matrix for Vector Space Model (Module 3)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.joined_corpus)

        # 2. Precompute statistics for Probabilistic BM25 (Module 3)
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_corpus]
        self.avg_dl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0

        # 3. Build Latent Semantic Space for Semantic Search (Module 5)
        n_components = min(4, self.tfidf_matrix.shape[1] - 1, self.N - 1)
        if n_components >= 1:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.semantic_doc_matrix = self.svd.fit_transform(self.tfidf_matrix)
        else:
            self.svd = None
            self.semantic_doc_matrix = None

    # -----------------------------------------------------------------
    # 1. VECTOR SPACE MODEL (TF-IDF & Cosine Similarity) - Module 3
    # -----------------------------------------------------------------
    def compute_vsm_tfidf(self, query: str) -> np.ndarray:
        """Calculates Cosine Similarity over TF-IDF vectors."""
        clean_query = " ".join(TextPreprocessor.preprocess(query))
        query_vec = self.tfidf_vectorizer.transform([clean_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return np.round(scores, 4)

    # -----------------------------------------------------------------
    # 2. PROBABILISTIC RETRIEVAL (Okapi BM25) - Module 3
    # -----------------------------------------------------------------
    def compute_bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Calculates normalized BM25 probabilistic relevance scores."""
        query_terms = TextPreprocessor.preprocess(query)
        raw_scores = np.zeros(self.N, dtype=float)

        for term in query_terms:
            # Document Frequency (df): number of documents containing the term
            df = sum(1 for doc_tokens in self.tokenized_corpus if term in doc_tokens)
            if df == 0:
                continue

            # BM25 Inverse Document Frequency (IDF)
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

            for idx, doc_tokens in enumerate(self.tokenized_corpus):
                tf = doc_tokens.count(term)
                if tf > 0:
                    dl = self.doc_lengths[idx]
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (dl / self.avg_dl))
                    raw_scores[idx] += idf * (numerator / denominator)

        # Normalize to [0, 1] so BM25 can be combined with VSM and Reliability scores
        max_score = np.max(raw_scores)
        norm_scores = raw_scores / max_score if max_score > 0 else raw_scores
        return np.round(norm_scores, 4)

    # -----------------------------------------------------------------
    # 3. SEMANTIC SEARCH (Latent Semantic Analysis) - Module 5
    # -----------------------------------------------------------------
    def compute_semantic_similarity(self, query: str) -> np.ndarray:
        """Matches queries and documents in reduced conceptual SVD space."""
        if self.svd is None or self.semantic_doc_matrix is None:
            return self.compute_vsm_tfidf(query)

        clean_query = " ".join(TextPreprocessor.preprocess(query))
        query_tfidf = self.tfidf_vectorizer.transform([clean_query])
        query_semantic = self.svd.transform(query_tfidf)

        sim = cosine_similarity(query_semantic, self.semantic_doc_matrix).flatten()
        # Clip negative SVD cosine values to 0.0
        sim = np.clip(sim, 0.0, 1.0)
        return np.round(sim, 4)

    # -----------------------------------------------------------------
    # 4. RELEVANCE FEEDBACK & QUERY EXPANSION (PRF) - Module 3
    # -----------------------------------------------------------------
    def expand_query_relevance_feedback(self, query: str, top_k_docs: int = 2, num_terms: int = 2) -> dict:
        """
        Uses Pseudo-Relevance Feedback (PRF):
        1. Ranks documents with initial BM25.
        2. Assumes top_k_docs are relevant feedback documents.
        3. Extracts highest mean TF-IDF terms from those top docs to expand the query.
        """
        initial_scores = self.compute_bm25(query)
        top_indices = np.argsort(initial_scores)[::-1][:top_k_docs]

        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        existing_query_stems = set(TextPreprocessor.preprocess(query))

        # Compute centroid of top feedback documents
        feedback_matrix = self.tfidf_matrix[top_indices].toarray()
        centroid_vector = np.mean(feedback_matrix, axis=0)
        ranked_term_indices = np.argsort(centroid_vector)[::-1]

        expansion_terms = []
        for idx in ranked_term_indices:
            candidate_term = feature_names[idx]
            if candidate_term not in existing_query_stems and centroid_vector[idx] > 0:
                expansion_terms.append(candidate_term)
                if len(expansion_terms) == num_terms:
                    break

        expanded_query = f"{query} {' '.join(expansion_terms)}".strip()
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "added_terms": expansion_terms,
            "feedback_doc_ids": [self.documents[i]["id"] for i in top_indices]
        }

    # -----------------------------------------------------------------
    # 5. PERSONALIZED SEARCH & COMBINED RETRIEVAL - Module 5
    # -----------------------------------------------------------------
    def retrieve_all_models(
        self,
        query: str,
        user_interests: list = None,
        use_query_expansion: bool = True
    ) -> tuple:
        """
        Runs Query Expansion, VSM (TF-IDF), BM25, Semantic Search, and Personalized Search.
        Returns per-document retrieval metrics and query expansion metadata.
        """
        if use_query_expansion:
            expansion_info = self.expand_query_relevance_feedback(query)
            active_query = expansion_info["expanded_query"]
        else:
            expansion_info = {
                "original_query": query,
                "expanded_query": query,
                "added_terms": [],
                "feedback_doc_ids": []
            }
            active_query = query

        vsm_scores = self.compute_vsm_tfidf(active_query)
        bm25_scores = self.compute_bm25(active_query)
        semantic_scores = self.compute_semantic_similarity(active_query)

        # Personalized Search boost (Module 5): match against user interest profile
        interest_query = " ".join(user_interests) if user_interests else ""
        if interest_query:
            personal_scores = self.compute_vsm_tfidf(interest_query)
        else:
            personal_scores = np.zeros(self.N)

        retrieval_results = []
        for idx, doc in enumerate(self.documents):
            # Combined IR Relevance Score:
            # 45% BM25 + 35% TF-IDF Cosine + 15% Semantic Search + 5% Personalization
            combined_relevance = (
                (0.45 * bm25_scores[idx]) +
                (0.35 * vsm_scores[idx]) +
                (0.15 * semantic_scores[idx]) +
                (0.05 * personal_scores[idx])
            )
            retrieval_results.append({
                "doc_id": doc["id"],
                "vsm_tfidf": float(vsm_scores[idx]),
                "bm25": float(bm25_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "personal_boost": float(personal_scores[idx]),
                "combined_relevance": round(float(combined_relevance), 4)
            })

        return retrieval_results, expansion_info


# =====================================================================
# QUICK TEST FOR STAGE 3
# =====================================================================
if __name__ == "__main__":
    try:
        from modules.stage1_ingestion_crawler import WebAndSocialCrawler
    except ImportError:
        from stage1_ingestion_crawler import WebAndSocialCrawler

    print("=== STAGE 3: RETRIEVAL MODELS, SEMANTIC & PERSONALIZED SEARCH TEST ===")
    crawler = WebAndSocialCrawler()
    docs, _ = crawler.crawl_and_collect("merit scholarship admission")

    engine = RetrievalEngine(docs)
    sample_query = "merit scholarship university admission"
    user_profile = ["undergraduate", "official", "financial", "portal"]

    results, exp_meta = engine.retrieve_all_models(
        query=sample_query,
        user_interests=user_profile,
        use_query_expansion=True
    )

    print(f"Original Query : {exp_meta['original_query']}")
    print(f"Expanded Query : {exp_meta['expanded_query']} (Added via PRF: {exp_meta['added_terms']})")
    print(f"Feedback Docs  : {exp_meta['feedback_doc_ids']}\n")

    print(f"{'Doc ID':<8} | {'BM25':<7} | {'TF-IDF':<7} | {'Semantic':<8} | {'Personal':<8} | {'Combined Rel.'}")
    print("-" * 68)
    for r in sorted(results, key=lambda x: x["combined_relevance"], reverse=True):
        print(f"{r['doc_id']:<8} | {r['bm25']:<7} | {r['vsm_tfidf']:<7} | "
              f"{r['semantic_score']:<8} | {r['personal_boost']:<8} | {r['combined_relevance']}")