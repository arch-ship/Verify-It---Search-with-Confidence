# 🔍 Verify It — Search with Confidence

**Verify It** is a hybrid **Information Retrieval (IR) and Text Analytics** search engine that retrieves live web documents and uploaded files while estimating their **reliability and source trustworthiness**. 

Standard keyword-based search algorithms (such as TF-IDF and BM25) evaluate relevance purely by term overlap, making them vulnerable to keyword-stuffed clickbait and unverified sources. **Verify It** bridges the gap between *relevant information* and *reliable information* by combining probabilistic/vector retrieval models with web graph link analysis (PageRank & HITS) and machine learning text classification (Naïve Bayes & KNN).

> **Academic Disclaimer:** Text and link topology alone cannot establish absolute objective truth. Verify It computes a **heuristic reliability estimate** based on defined IR and Text Analytics indicators rather than acting as a definitive fact-checking oracle.

---

## ✨ Key Features & Architecture

The system is architected across **5 modular stages**, covering end-to-end Information Retrieval and Text Mining concepts:

### 1. Multi-Format Ingestion, Cross-Language IR & Live Web Crawling (`stage1_ingestion_crawler.py`)
* **Live Web & Encyclopedia Crawler:** Dynamically fetches real-time web results via DuckDuckGo HTML scraping and the Wikipedia OpenSearch API for any user query, constructing a directed domain hyperlink graph on the fly.
* **Multi-Format Document Parser:** Extracts text and metadata from unstructured (`.txt`, `.pdf`, `.docx`) and semi-structured (`.xml`) files.
* **Cross-Language Information Retrieval (CLIR) & Spell Checking:** Translates non-English queries (e.g., Hindi/Spanish) into normalized English search representations and provides Google-style **"Did you mean?"** spelling corrections using character $n$-gram edit distance.

### 2. Text Preprocessing & Multi-Index Construction (`stage2_indexer.py`)
* **Preprocessing Pipeline:** Regex tokenization, stop-word removal, dictionary-backed lemmatization, and suffix-stripping stemming.
* **Index Structures:**
  * **Inverted File Index:** Maps stemmed terms to sorted document posting lists.
  * **Positional Index:** Tracks exact word offsets within documents to support exact phrase queries.
  * **Character $n$-gram Index:** Builds 3-gram character mappings for spell/wildcard tolerance.
  * **Index Compression:** Applies **Gap (Delta) Encoding** to compress numeric posting lists.

### 3. Multi-Model Information Retrieval & Query Expansion (`stage3_retrieval.py`)
* **Vector Space Model (VSM):** Computes TF-IDF weighting and Cosine Similarity.
* **Probabilistic Retrieval (Okapi BM25):** Scores documents with document-length normalization ($k_1 = 1.5, b = 0.75$).
* **Semantic Search:** Projects term-document matrices into a low-dimensional conceptual space using **Latent Semantic Analysis (Truncated SVD)**.
* **Relevance Feedback & Query Expansion:** Uses **Pseudo-Relevance Feedback (PRF)** to extract high-weight centroid terms from top initial hits and expand the user's query automatically.

### 4. Reliability Mining, Link Analysis & Clustering (`stage4_reliability.py`)
* **Web Graph Link Analysis:** Computes **PageRank** and **HITS (Hubs & Authorities)** scores over the crawled domain graph.
* **Spam & Clickbait Detection:** Trains **Multinomial Naïve Bayes** and **k-Nearest Neighbors (KNN)** classifiers alongside sensationalist pattern penalties.
* **Cross-Document Claim Consensus:** Evaluates repeated factual claims across authoritative sources via n-gram cosine similarity.
* **Document Clustering:** Groups retrieved documents into topical/quality clusters using **k-Means Clustering**.

### 5. Hybrid Ranking, Question Answering & Evaluation (`stage5_ranking_qa.py`)
* **Tunable Hybrid Ranking & Trust Sorting:** Combines IR Relevance and Reliability using a configurable weight ($\alpha$), automatically sorting search results in **descending order of verified source trust**.
* **Extractive Question Answering (QA):** Identifies and extracts the highest-overlap answer sentence strictly from the **#1 most reliable source**.
* **IR Performance Evaluation:** Benchmarks Standard BM25 Ranking against Verify-It Hybrid Ranking using **Precision@K**, **Recall@K**, **F1-Score@K**, and **Average Precision (AP)**.

---

## 🎨 Frontend & UI/UX Highlights
* **Google-Style Interface:** Starts with a centered search experience that docks smoothly to the top upon query submission.
* **Auto-Expanding Textarea:** Dynamically grows vertically as users paste long paragraphs or claims to verify.
* **Claude/Gemini-Style Source Verification Stream:** Features a sequential 3-dot animation (`• • •`) as the system visits each crawled source, displaying Google's classic *"Searched about X results in (Y.YY seconds)"* statistic upon completion.
* **3D Flip Cards with Motion Blur:** Medium-sized vertical rectangular cards with rounded corners (`rounded-3xl`) that pop up and smoothly flip in 3D space with horizontal motion blur on hover to reveal underlying IR diagnostics (BM25, TF-IDF, PageRank, HITS, Naive Bayes).
* **AI Glow Search Button:** A mouse-tracking radial spotlight aura that follows your cursor across the search button.

---

## 📁 Project Structure

```text
Verify-It---Search-with-Confidence/
├── api/
│   └── index.py                       # FastAPI Serverless Backend (Vercel entry point)
├── data/
│   └── sample_upload.xml              # Sample semi-structured XML document for testing
├── modules/
│   ├── __init__.py
│   ├── stage1_ingestion_crawler.py    # File parser, CLIR, Spell Check & Live Web Crawler
│   ├── stage2_indexer.py              # Stemmer, Inverted, Positional & Compressed Indexes
│   ├── stage3_retrieval.py            # TF-IDF, BM25, Semantic SVD & PRF Query Expansion
│   ├── stage4_reliability.py          # PageRank, HITS, Naive Bayes, KNN & k-Means
│   └── stage5_ranking_qa.py           # Hybrid Ranking, Extractive QA & Evaluation Metrics
├── public/
│   └── index.html                     # Google-style Frontend UI with 3D Flip Cards & 3-dot stream
├── vercel.json                        # Vercel Serverless deployment routing config
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation