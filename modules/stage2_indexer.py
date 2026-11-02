import re
from collections import defaultdict


# =====================================================================
# 1. TEXT PREPROCESSING (Tokenization, Stop-words, Stemming, Lemmatization)
#    Syllabus Module 2: Text Processing and Indexing
# =====================================================================
class TextPreprocessor:
    """
    Handles Tokenization, Normalization, Stop-word removal, 
    Lemmatization, and Stemming without heavy external NLP model downloads.
    """

    STOP_WORDS = {
        "the", "is", "in", "to", "for", "and", "of", "a", "an", "on",
        "with", "has", "at", "by", "this", "will", "be", "have", "been",
        "are", "from", "or", "that", "as", "it", "not", "do", "us", "now"
    }

    # Dictionary-based Lemmatization for common irregular academic/domain words
    LEMMA_DICT = {
        "universities": "university",
        "studies": "study",
        "criteria": "criterion",
        "policies": "policy",
        "children": "child",
        "better": "good",
        "best": "good",
        "applied": "apply",
        "applying": "apply",
        "applications": "application",
        "examinations": "examination"
    }

    @classmethod
    def tokenize(cls, text: str) -> list:
        """Extracts alphanumeric lowercase tokens."""
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    @classmethod
    def remove_stopwords(cls, tokens: list) -> list:
        """Filters out common stop words."""
        return [t for t in tokens if t not in cls.STOP_WORDS]

    @classmethod
    def lemmatize(cls, word: str) -> str:
        """Maps words to their dictionary base form (lemma)."""
        if word in cls.LEMMA_DICT:
            return cls.LEMMA_DICT[word]
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        return word

    @classmethod
    def stem(cls, word: str) -> str:
        """Porter-style suffix stripping stemmer."""
        for suffix in ["ation", "ments", "ment", "ing", "ed", "es", "s", "ly"]:
            if word.endswith(suffix) and (len(word) - len(suffix)) >= 3:
                return word[:-len(suffix)]
        return word

    @classmethod
    def preprocess(cls, text: str, use_stemming: bool = True) -> list:
        """Full preprocessing pipeline: Tokenize -> Stop-words -> Lemmatize -> Stem."""
        tokens = cls.tokenize(text)
        filtered = cls.remove_stopwords(tokens)
        lemmas = [cls.lemmatize(t) for t in filtered]
        if use_stemming:
            return [cls.stem(t) for t in lemmas]
        return lemmas


# =====================================================================
# 2. MULTI-INDEX BUILDER (Inverted, Positional, n-gram & Compression)
#    Syllabus Module 1 & Module 2: Index Construction & Compression
# =====================================================================
class DocumentIndexer:
    """
    Constructs and queries:
    1. Inverted File Index: term -> sorted list of doc_ids
    2. Positional Index   : term -> {doc_id: [word_positions]}
    3. Character n-gram Index: n-gram -> set of terms (for wildcard/spell tolerance)
    4. Compressed Index   : Gap-encoded (Delta-encoded) posting lists
    """

    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size
        self.inverted_index = defaultdict(set)
        self.positional_index = defaultdict(lambda: defaultdict(list))
        self.ngram_index = defaultdict(set)
        self.compressed_index = {}
        self.doc_map = {}

    def build_indexes(self, documents: list):
        """Builds Inverted, Positional, n-gram, and Compressed indexes from documents."""
        self.inverted_index.clear()
        self.positional_index.clear()
        self.ngram_index.clear()
        self.compressed_index.clear()
        self.doc_map = {idx + 1: doc["id"] for idx, doc in enumerate(documents)}
        reverse_doc_map = {doc["id"]: idx + 1 for idx, doc in enumerate(documents)}

        for doc in documents:
            doc_id = doc["id"]
            full_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = TextPreprocessor.preprocess(full_text)

            for pos, term in enumerate(tokens):
                # 1. Inverted Index
                self.inverted_index[term].add(doc_id)

                # 2. Positional Index
                self.positional_index[term][doc_id].append(pos)

                # 3. Character n-gram Index (e.g., $sch$ -> sch, cho, hol...)
                padded_term = f"${term}$"
                for i in range(len(padded_term) - self.ngram_size + 1):
                    ngram = padded_term[i:i + self.ngram_size]
                    self.ngram_index[ngram].add(term)

        # Sort inverted index posting lists and build Gap-Encoded Compressed Index
        for term, doc_ids in self.inverted_index.items():
            sorted_ids = sorted(doc_ids)
            self.inverted_index[term] = sorted_ids

            # Convert doc_ids to integer IDs for Gap Compression (Delta Encoding)
            numeric_postings = sorted(reverse_doc_map[d] for d in sorted_ids)
            self.compressed_index[term] = self._gap_encode(numeric_postings)

    @staticmethod
    def _gap_encode(sorted_numbers: list) -> list:
        """
        Module 2: Index Compression Concept (Gap / Delta Encoding).
        Stores the first doc_id and only the gaps between subsequent doc_ids.
        Example: [1, 2, 5, 6] -> [1, 1, 3, 1]
        """
        if not sorted_numbers:
            return []
        gaps = [sorted_numbers[0]]
        for i in range(1, len(sorted_numbers)):
            gaps.append(sorted_numbers[i] - sorted_numbers[i - 1])
        return gaps

    # -----------------------------------------------------------------
    # BOOLEAN RETRIEVAL & POSITIONAL PHRASE QUERIES (Module 1 & 2)
    # -----------------------------------------------------------------
    def boolean_query(self, term1: str, operator: str, term2: str) -> list:
        """Supports AND, OR, NOT Boolean Retrieval queries."""
        t1_list = TextPreprocessor.preprocess(term1)
        t2_list = TextPreprocessor.preprocess(term2)
        t1 = t1_list[0] if t1_list else ""
        t2 = t2_list[0] if t2_list else ""

        set1 = set(self.inverted_index.get(t1, []))
        set2 = set(self.inverted_index.get(t2, []))

        op = operator.upper().strip()
        if op == "AND":
            return sorted(set1.intersection(set2))
        elif op == "OR":
            return sorted(set1.union(set2))
        elif op == "NOT":
            return sorted(set1.difference(set2))
        else:
            raise ValueError(f"Unsupported Boolean operator: {operator}")

    def phrase_query(self, phrase: str) -> list:
        """Uses the Positional Index to find documents where terms appear consecutively."""
        terms = TextPreprocessor.preprocess(phrase)
        if not terms:
            return []
        if len(terms) == 1:
            return list(self.inverted_index.get(terms[0], []))

        # Candidate documents must contain all terms in the phrase
        candidate_docs = set(self.inverted_index.get(terms[0], []))
        for t in terms[1:]:
            candidate_docs = candidate_docs.intersection(self.inverted_index.get(t, []))

        matching_docs = []
        for doc_id in sorted(candidate_docs):
            first_term_positions = self.positional_index[terms[0]][doc_id]
            for start_pos in first_term_positions:
                # Check if subsequent terms appear at start_pos + 1, start_pos + 2, etc.
                if all(
                    (start_pos + offset) in self.positional_index[terms[offset]][doc_id]
                    for offset in range(1, len(terms))
                ):
                    matching_docs.append(doc_id)
                    break

        return matching_docs


# =====================================================================
# QUICK TEST FOR STAGE 2
# =====================================================================
if __name__ == "__main__":
    # Import Stage 1 to test pipeline continuity
    try:
        from modules.stage1_ingestion_crawler import WebAndSocialCrawler
    except ImportError:
        from stage1_ingestion_crawler import WebAndSocialCrawler

    print("=== STAGE 2: PREPROCESSING & MULTI-INDEXER TEST ===")
    crawler = WebAndSocialCrawler()
    docs, _ = crawler.crawl_and_collect("merit scholarship admission")

    indexer = DocumentIndexer(ngram_size=3)
    indexer.build_indexes(docs)

    sample_term = TextPreprocessor.preprocess("scholarship")[0]
    print(f"1. Stemmed Term for 'scholarship' : '{sample_term}'")
    print(f"2. Inverted Index Postings        : {indexer.inverted_index[sample_term]}")
    print(f"3. Positional Index Postings      : {dict(indexer.positional_index[sample_term])}")
    print(f"4. Gap-Compressed Posting List    : {indexer.compressed_index[sample_term]}")
    print(f"5. 3-gram Index for '$sc'         : {indexer.ngram_index['$sc']}")
    print(f"6. Boolean Query (scholarship AND examination): "
          f"{indexer.boolean_query('scholarship', 'AND', 'examination')}")
    print(f"7. Positional Phrase Query ('merit scholarship'): "
          f"{indexer.phrase_query('merit scholarship')}")