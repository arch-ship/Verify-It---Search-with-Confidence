import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlparse

# Optional file/translation libraries with safe fallbacks
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None


# =====================================================================
# 1. MULTI-FORMAT FILE PARSER (Unstructured, Semi-Structured & XML)
#    Syllabus Module 1 (Data Types) & Module 5 (XML Search)
# =====================================================================
class DocumentParser:
    """Extracts text and metadata from uploaded TXT, PDF, DOCX, or XML files."""

    @staticmethod
    def parse_file(file_path: str) -> dict:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".xml":
            return DocumentParser._parse_xml(file_path)
        elif ext == ".pdf":
            return DocumentParser._parse_pdf(file_path)
        elif ext == ".docx":
            return DocumentParser._parse_docx(file_path)
        else:
            return DocumentParser._parse_txt(file_path)

    @staticmethod
    def _parse_xml(file_path: str) -> dict:
        """Module 5: XML Search & Semi-Structured Document Parsing."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        return {
            "id": "UPLOAD_XML",
            "title": root.findtext("title", default="Uploaded XML Document"),
            "source": root.findtext("source", default="user-upload.local"),
            "date": root.findtext("date", default=datetime.now().strftime("%Y-%m-%d")),
            "text": root.findtext("body", default=" ".join(root.itertext())),
            "type": "xml_structured"
        }

    @staticmethod
    def _parse_pdf(file_path: str) -> dict:
        if not PdfReader:
            raise ImportError("Install pypdf (`pip install pypdf`) to read PDF files.")
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return DocumentParser._wrap_unstructured(file_path, text)

    @staticmethod
    def _parse_docx(file_path: str) -> dict:
        if not docx:
            raise ImportError("Install python-docx (`pip install python-docx`) to read DOCX files.")
        doc = docx.Document(file_path)
        text = " ".join(p.text for p in doc.paragraphs)
        return DocumentParser._wrap_unstructured(file_path, text)

    @staticmethod
    def _parse_txt(file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return DocumentParser._wrap_unstructured(file_path, text)

    @staticmethod
    def _wrap_unstructured(file_path: str, text: str) -> dict:
        return {
            "id": "UPLOAD_DOC",
            "title": os.path.basename(file_path),
            "source": "user-upload.local",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "text": text.strip(),
            "type": "unstructured"
        }


# =====================================================================
# 2. QUERY FORMULATION & CROSS-LANGUAGE IR (CLIR)
#    Syllabus Module 3 (Query Processing) & Module 5 (Cross-Language IR)
# =====================================================================
class QueryFormulator:
    """Translates non-English inputs (CLIR) and extracts key search queries from long texts/files."""

    # Offline fallback dictionary for Hindi/Spanish terms if no internet connection
    OFFLINE_CLIR_DICT = {
        "छात्रवृत्ति": "scholarship",
        "प्रवेश": "admission",
        "विश्वविद्यालय": "university",
        "नीति": "policy",
        "परीक्षा": "examination",
        "beca": "scholarship",
        "universidad": "university"
    }

    @classmethod
    def translate_to_english(cls, text: str) -> str:
        """Module 5: Cross-Language Information Retrieval (CLIR)."""
        # Check offline dictionary first for deterministic demo reliability
        words = text.split()
        translated_words = [cls.OFFLINE_CLIR_DICT.get(w, w) for w in words]
        candidate = " ".join(translated_words)

        # If non-ASCII characters remain and GoogleTranslator is installed, translate live
        if not candidate.isascii() and GoogleTranslator is not None:
            try:
                return GoogleTranslator(source="auto", target="en").translate(text)
            except Exception:
                pass
        return candidate

    @classmethod
    def formulate_query_from_input(cls, raw_text: str, max_terms: int = 6) -> str:
        """Extracts core search keywords if the user uploads a large document or paragraph."""
        english_text = cls.translate_to_english(raw_text)
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', english_text.lower())
        
        stop_words = {
            "this", "that", "with", "from", "have", "will", "your", 
            "about", "into", "their", "there", "which", "would", "could"
        }
        keywords = [t for t in tokens if t not in stop_words]
        
        # Rank by term frequency inside the uploaded file/input to formulate query
        freq = {}
        for k in keywords:
            freq[k] = freq.get(k, 0) + 1
        top_keywords = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:max_terms]
        
        return " ".join(top_keywords)


# =====================================================================
# 3. WEB CRAWLER & SOCIAL MEDIA COLLECTOR
#    Syllabus Module 5 (Web Crawling, Web Search Engines, Social Media IR)
# =====================================================================
class WebAndSocialCrawler:
    """
    Collects Web Pages and Social Media posts along with their hyperlink graph
    (outlinks) so Stage 4 can compute PageRank and HITS (Hubs & Authorities).
    """

    def __init__(self):
        # Curated Web + Social Media index for reproducible offline/online testing
        self.knowledge_web = [
            {
                "id": "WEB_1",
                "title": "National Merit Scholarship 2026 Official Guidelines",
                "url": "https://education.gov.in/scholarships-2026",
                "source": "education.gov.in",
                "date": "2026-01-15",
                "type": "official_web",
                "text": "The National Merit Scholarship provides financial aid to undergraduate students. Applicants must have a minimum 85 percent score in higher secondary examinations. Official portal applications close in October.",
                "outlinks": ["jiit.ac.in", "ugc.ac.in"],
                "label_reliable": 1,
                "is_relevant": 1
            },
            {
                "id": "WEB_2",
                "title": "University Admission and Merit Scholarship Requirements",
                "url": "https://jiit.ac.in/admissions/scholarship",
                "source": "jiit.ac.in",
                "date": "2025-11-20",
                "type": "academic_web",
                "text": "University admission requires valid entrance examination scores. Applicants must have a minimum 85 percent score for direct merit scholarship eligibility and submit verified transcripts.",
                "outlinks": ["education.gov.in", "ugc.ac.in"],
                "label_reliable": 1,
                "is_relevant": 1
            },
            {
                "id": "WEB_3",
                "title": "100% Guaranteed Free Scholarship Trick! Apply Now!!",
                "url": "http://free-money-blog.xyz/instant-scholarship",
                "source": "free-money-blog.xyz",
                "date": "2023-05-10",
                "type": "blog_web",
                "text": "Click here for guaranteed scholarship money! No exams needed, instant approval for university admission and merit scholarship. Secret loophole cash prize revealed!!",
                "outlinks": ["fast-degree-scam.net"],
                "label_reliable": 0,
                "is_relevant": 0
            },
            {
                "id": "SOC_1",
                "title": "Social Media Thread: Verified Scholarship Update",
                "url": "https://social-edu.org/post/9921",
                "source": "social-edu.org",
                "date": "2026-02-10",
                "type": "social_media",
                "text": "Reminder for students: #Scholarship2026 applications on education.gov.in require 85 percent marks in 12th grade. Do not fall for instant approval scam links circulating online!",
                "outlinks": ["education.gov.in", "jiit.ac.in"],
                "label_reliable": 1,
                "is_relevant": 1
            },
            {
                "id": "SOC_2",
                "title": "Viral Post: Free University Admission Without Exams!",
                "url": "http://fast-degree-scam.net/viral-post",
                "source": "fast-degree-scam.net",
                "date": "2022-03-01",
                "type": "social_media",
                "text": "DM us now! Get instant university admission and free scholarship cash without entrance examinations! 100% guaranteed approval trick for students.",
                "outlinks": ["free-money-blog.xyz"],
                "label_reliable": 0,
                "is_relevant": 0
            },
            {
                "id": "WEB_4",
                "title": "Old Archive: 2019 Scholarship Rules and Policy",
                "url": "https://ugc.ac.in/archive/2019-policy",
                "source": "ugc.ac.in",
                "date": "2019-08-12",
                "type": "official_web",
                "text": "Under the 2019 policy, students asking about the merit scholarship needed 75 percent marks. Please check newer official circulars for 2026 updated dates and criteria.",
                "outlinks": ["education.gov.in"],
                "label_reliable": 1,
                "is_relevant": 1
            }
        ]

    def crawl_and_collect(self, formulated_query: str, uploaded_doc: dict = None):
        """
        Simulates crawling web + social media sources and builds the hyperlink graph
        needed for PageRank & HITS in Stage 4.
        """
        collected_docs = list(self.knowledge_web)
        if uploaded_doc:
            uploaded_doc["outlinks"] = ["education.gov.in"]
            uploaded_doc["label_reliable"] = 1
            uploaded_doc["is_relevant"] = 1
            collected_docs.insert(0, uploaded_doc)

        # Build Directed Link Graph: {source_domain: [linked_domains]}
        link_graph = {}
        for doc in collected_docs:
            src = doc["source"]
            if src not in link_graph:
                link_graph[src] = []
            for target in doc.get("outlinks", []):
                if target not in link_graph[src]:
                    link_graph[src].append(target)
                if target not in link_graph:
                    link_graph[target] = []

        return collected_docs, link_graph


# =====================================================================
# QUICK TEST FOR STAGE 1
# =====================================================================
if __name__ == "__main__":
    print("=== STAGE 1: INGESTION, CLIR & WEB CRAWLER TEST ===")
    
    # Test 1: Cross-Language Query Formulation (Hindi + English mix)
    raw_user_input = "छात्रवृत्ति विश्वविद्यालय प्रवेश merit scholarship requirements"
    formulated_query = QueryFormulator.formulate_query_from_input(raw_user_input)
    print(f"Raw User Input   : {raw_user_input}")
    print(f"Formulated Query : {formulated_query}")

    # Test 2: Crawl Web & Social Media Sources + Build Hyperlink Graph
    crawler = WebAndSocialCrawler()
    docs, web_graph = crawler.crawl_and_collect(formulated_query)
    print(f"\nCollected {len(docs)} documents (Web, Social Media, Archives).")
    print("Constructed Hyperlink Graph for PageRank/HITS:")
    for domain, targets in web_graph.items():
        print(f"  {domain:<22} -> {targets}")