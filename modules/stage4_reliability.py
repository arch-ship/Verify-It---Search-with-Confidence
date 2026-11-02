import re
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================================
# 1. WEB LINK ANALYSIS: PAGERANK & HITS (Hubs and Authorities)
#    Syllabus Module 5: Ranking Algorithms, PageRank and HITS
# =====================================================================
class WebGraphAnalyzer:
    """Computes PageRank and HITS (Hub & Authority) scores on the domain link graph."""

    @staticmethod
    def compute_pagerank(link_graph: dict, damping: float = 0.85, iterations: int = 25) -> dict:
        nodes = list(link_graph.keys())
        n = len(nodes)
        if n == 0:
            return {}
        ranks = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_ranks = {}
            for node in nodes:
                incoming_sum = 0.0
                for other_node, outlinks in link_graph.items():
                    if node in outlinks and len(outlinks) > 0:
                        incoming_sum += ranks[other_node] / len(outlinks)
                new_ranks[node] = ((1.0 - damping) / n) + (damping * incoming_sum)
            ranks = new_ranks

        max_pr = max(ranks.values()) if ranks else 1.0
        return {k: round(v / max_pr, 4) for k, v in ranks.items()}

    @staticmethod
    def compute_hits(link_graph: dict, iterations: int = 25) -> tuple:
        """
        Module 5: HITS Algorithm (Hyperlink-Induced Topic Search).
        Computes Authority scores (trusted sources cited by many hubs)
        and Hub scores (pages that link to many authorities).
        """
        nodes = list(link_graph.keys())
        hubs = {node: 1.0 for node in nodes}
        authorities = {node: 1.0 for node in nodes}

        for _ in range(iterations):
            # Authority update: sum of hub scores of incoming links
            new_auth = {node: 0.0 for node in nodes}
            for node in nodes:
                for other_node, outlinks in link_graph.items():
                    if node in outlinks:
                        new_auth[node] += hubs[other_node]

            # Normalize authorities
            norm_a = math.sqrt(sum(v ** 2 for v in new_auth.values())) or 1.0
            authorities = {k: v / norm_a for k, v in new_auth.items()}

            # Hub update: sum of authority scores of outgoing links
            new_hubs = {node: 0.0 for node in nodes}
            for node, outlinks in link_graph.items():
                new_hubs[node] = sum(authorities.get(target, 0.0) for target in outlinks)

            # Normalize hubs
            norm_h = math.sqrt(sum(v ** 2 for v in new_hubs.values())) or 1.0
            hubs = {k: v / norm_h for k, v in new_hubs.items()}

        max_auth = max(authorities.values()) if authorities else 1.0
        max_hub = max(hubs.values()) if hubs else 1.0
        norm_authorities = {k: round(v / max_auth, 4) for k, v in authorities.items()}
        norm_hubs = {k: round(v / max_hub, 4) for k, v in hubs.items()}
        return norm_authorities, norm_hubs


import math


# =====================================================================
# 2. TEXT MINING & RELIABILITY ENGINE
#    Syllabus Module 4: Naive Bayes, KNN, Spam Detection, k-Means Clustering
#    Synopsis Features: Source Info, Dates, Repeated Claims, Text Patterns
# =====================================================================
class ReliabilityAnalyzer:
    """
    Estimates document reliability by combining:
    1. Source Authority (PageRank + HITS Authority + Domain TLD verification)
    2. Text Classification / Spam Detection (Multinomial Naive Bayes + KNN)
    3. Repeated Claim Verification (Cross-document factual consensus similarity)
    4. Publication Date Recency
    5. k-Means Document Clustering
    """

    def __init__(self, documents: list, link_graph: dict):
        self.documents = documents
        self.link_graph = link_graph

        # 1. Compute PageRank and HITS scores
        self.pagerank = WebGraphAnalyzer.compute_pagerank(link_graph)
        self.hits_authority, self.hits_hubs = WebGraphAnalyzer.compute_hits(link_graph)

        # 2. Train Naive Bayes & KNN Classifiers for Spam/Reliability Detection (Module 4)
        self.vectorizer = CountVectorizer(stop_words="english")
        corpus = [f"{d.get('title', '')} {d.get('text', '')}" for d in documents]
        self.X_counts = self.vectorizer.fit_transform(corpus)
        y_labels = [d.get("label_reliable", 1) for d in documents]

        # Multinomial Naive Bayes Classifier
        self.nb_classifier = MultinomialNB()
        self.nb_classifier.fit(self.X_counts, y_labels)

        # k-Nearest Neighbors (KNN) Classifier
        k_neighbors = min(3, len(documents))
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric="cosine")
        self.knn_classifier.fit(self.X_counts, y_labels)

    def _compute_repeated_claim_scores(self) -> np.ndarray:
        """
        Synopsis Indicator: Repeated Claims.
        Checks whether core numeric/factual claims (e.g., '85 percent', 'october')
        are corroborated across multiple authoritative documents.
        """
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [d.get("text", "") for d in self.documents]
        matrix = tfidf.fit_transform(corpus)
        sim_matrix = cosine_similarity(matrix)

        # Weight consensus by whether corroborating documents are from official/academic domains
        authority_weights = np.array([
            1.0 if d.get("source", "").endswith((".gov.in", ".ac.in", ".org")) else 0.2
            for d in self.documents
        ])

        consensus_scores = []
        for i in range(len(self.documents)):
            # Similarity with other documents weighted by their domain authority
            other_sims = [
                sim_matrix[i][j] * authority_weights[j]
                for j in range(len(self.documents)) if i != j
            ]
            score = float(np.mean(other_sims)) if other_sims else 0.0
            consensus_scores.append(score)

        max_c = max(consensus_scores) if consensus_scores and max(consensus_scores) > 0 else 1.0
        return np.round(np.array(consensus_scores) / max_c, 4)

    def run_kmeans_clustering(self, n_clusters: int = 2) -> dict:
        """Module 4: k-Means Clustering to organize documents into topical/quality groups."""
        k = min(n_clusters, len(self.documents))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X_counts)

        clusters = defaultdict(list)
        doc_cluster_map = {}
        for idx, cluster_id in enumerate(labels):
            doc_id = self.documents[idx]["id"]
            clusters[int(cluster_id)].append(doc_id)
            doc_cluster_map[doc_id] = int(cluster_id)

        return {"clusters": dict(clusters), "doc_cluster_map": doc_cluster_map}

    def evaluate_all_documents(self, current_year: int = 2026) -> tuple:
        """Calculates comprehensive reliability breakdown for every document."""
        nb_probs = self.nb_classifier.predict_proba(self.X_counts)[:, 1]
        knn_probs = self.knn_classifier.predict_proba(self.X_counts)[:, 1]
        claim_scores = self._compute_repeated_claim_scores()
        cluster_info = self.run_kmeans_clustering(n_clusters=2)

        reliability_reports = []
        for idx, doc in enumerate(self.documents):
            src = doc.get("source", "")
            pr = self.pagerank.get(src, 0.1)
            auth = self.hits_authority.get(src, 0.1)
            hub = self.hits_hubs.get(src, 0.1)

            # 1. Web Link & Source Authority Score (PageRank + HITS Authority)
            link_authority = round((0.5 * pr) + (0.5 * auth), 4)

            # 2. Text Pattern / Spam Filter Score (Naive Bayes + KNN + Clickbait Penalty)
            full_text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            clickbait_triggers = len(re.findall(r'\b(guaranteed|trick|secret|loophole|instant|scam|cash|100%)\b', full_text))
            heuristic_trust = max(0.0, 1.0 - (clickbait_triggers * 0.25) - (full_text.count("!") * 0.1))
            text_pattern_score = round((0.45 * float(nb_probs[idx])) + (0.30 * float(knn_probs[idx])) + (0.25 * heuristic_trust), 4)

            # 3. Publication Date Recency Score
            try:
                pub_year = datetime.strptime(doc.get("date", "2026-01-01"), "%Y-%m-%d").year
            except ValueError:
                pub_year = current_year
            age = max(0, current_year - pub_year)
            recency_score = round(max(0.15, 1.0 - (age * 0.12)), 4)

            # 4. Repeated Claims Consensus Score
            claim_consensus = float(claim_scores[idx])

            # Overall Reliability Estimate:
            # 35% Link Authority (PageRank/HITS) + 35% Text Mining (NB/KNN) + 15% Claim Consensus + 15% Recency
            total_reliability = (
                (0.35 * link_authority) +
                (0.35 * text_pattern_score) +
                (0.15 * claim_consensus) +
                (0.15 * recency_score)
            )

            reliability_reports.append({
                "doc_id": doc["id"],
                "source": src,
                "pagerank": pr,
                "hits_authority": auth,
                "hits_hub": hub,
                "nb_trust_prob": round(float(nb_probs[idx]), 4),
                "knn_trust_prob": round(float(knn_probs[idx]), 4),
                "text_pattern_score": text_pattern_score,
                "claim_consensus": claim_consensus,
                "recency_score": recency_score,
                "cluster_id": cluster_info["doc_cluster_map"][doc["id"]],
                "total_reliability": round(float(total_reliability), 4)
            })

        return reliability_reports, cluster_info["clusters"]


# =====================================================================
# QUICK TEST FOR STAGE 4
# =====================================================================
if __name__ == "__main__":
    try:
        from modules.stage1_ingestion_crawler import WebAndSocialCrawler
    except ImportError:
        from stage1_ingestion_crawler import WebAndSocialCrawler

    print("=== STAGE 4: PAGERANK, HITS, NAIVE BAYES, KNN & K-MEANS RELIABILITY TEST ===")
    crawler = WebAndSocialCrawler()
    docs, link_graph = crawler.crawl_and_collect("merit scholarship admission")

    analyzer = ReliabilityAnalyzer(docs, link_graph)
    reports, clusters = analyzer.evaluate_all_documents()

    print(f"k-Means Document Clusters (k=2): {clusters}\n")
    print(f"{'Doc ID':<8} | {'PRank':<6} | {'HITS_A':<6} | {'NB_Prb':<6} | {'KNN_Prb':<7} | {'Claims':<6} | {'Recency':<7} | {'RELIABILITY'}")
    print("-" * 86)
    for r in sorted(reports, key=lambda x: x["total_reliability"], reverse=True):
        print(f"{r['doc_id']:<8} | {r['pagerank']:<6} | {r['hits_authority']:<6} | "
              f"{r['nb_trust_prob']:<6} | {r['knn_trust_prob']:<7} | "
              f"{r['claim_consensus']:<6} | {r['recency_score']:<7} | {r['total_reliability']}")