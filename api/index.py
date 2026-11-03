import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from modules.stage1_ingestion_crawler import (
    DocumentParser,
    QueryFormulator,
    WebAndSocialCrawler
)
from modules.stage2_indexer import DocumentIndexer, TextPreprocessor
from modules.stage5_ranking_qa import HybridRankingQAEngine

app = FastAPI(title="Verify It API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h2>Verify It API is running. Place index.html in /public.</h2>"


@app.post("/api/search")
async def search_endpoint(
    query: str = Form(""),
    alpha: float = Form(0.50),
    use_prf: bool = Form(True),
    interests: str = Form("official, verified, academic, research"),
    file: UploadFile = File(None)
):
    uploaded_doc = None
    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            uploaded_doc = DocumentParser.parse_file(tmp_path, original_name=file.filename)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if uploaded_doc and not query.strip():
        active_query = QueryFormulator.formulate_query_from_input(uploaded_doc["text"])
    else:
        active_query = QueryFormulator.formulate_query_from_input(query.strip() or "information retrieval")

    crawler = WebAndSocialCrawler()
    docs, link_graph = crawler.crawl_and_collect(active_query, uploaded_doc=uploaded_doc)

    # Build dynamic vocabulary from crawled titles/texts for "Did you mean?" spell suggestion
    dynamic_vocab = set()
    for d in docs:
        dynamic_vocab.update(re.findall(r'\b[a-zA-Z]{4,}\b', f"{d.get('title', '')} {d.get('text', '')}".lower()))
    suggested_query = QueryFormulator.suggest_query_correction(active_query, dynamic_vocab=dynamic_vocab)

    # Compute Lemmatized & Stemmed Tokens for display
    raw_tokens = TextPreprocessor.remove_stopwords(TextPreprocessor.tokenize(active_query))
    lemmatized_tokens = [TextPreprocessor.lemmatize(t) for t in raw_tokens]
    stemmed_tokens = [TextPreprocessor.stem(t) for t in lemmatized_tokens]

    indexer = DocumentIndexer(ngram_size=3)
    indexer.build_indexes(docs)

    user_profile = [w.strip() for w in interests.split(",") if w.strip()]
    engine = HybridRankingQAEngine(docs, link_graph)
    pipeline_out = engine.execute_search_pipeline(
        query=active_query,
        user_interests=user_profile,
        alpha=float(alpha),
        use_query_expansion=bool(use_prf)
    )

    sample_stem = stemmed_tokens[0] if stemmed_tokens else ""

    return JSONResponse({
        "raw_query": query,
        "active_query": active_query,
        "suggested_query": suggested_query,
        "lemmatized_tokens": lemmatized_tokens,
        "stemmed_tokens": stemmed_tokens,
        "expanded_query": pipeline_out["query_metadata"]["expanded_query"],
        "added_terms": pipeline_out["query_metadata"]["added_terms"],
        "direct_answer": pipeline_out["extracted_answer"],
        "hybrid_results": pipeline_out["hybrid_ranking"],
        "standard_results": pipeline_out["standard_ranking"],
        "evaluation": pipeline_out["evaluation"],
        "clusters": pipeline_out["clusters"],
        "index_preview": {
            "stem": sample_stem,
            "inverted": indexer.inverted_index.get(sample_stem, []),
            "compressed_gaps": indexer.compressed_index.get(sample_stem, []),
            "positional": dict(indexer.positional_index.get(sample_stem, {}))
        }
    })