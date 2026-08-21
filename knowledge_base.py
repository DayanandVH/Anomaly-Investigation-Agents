"""
RAG knowledge base for the anomaly-investigation agent.

Design note: the real Kaggle credit-card dataset's V1..V28 columns are
anonymized PCA components with NO real-world meaning (that's why the bank
released them that way). So we can't write case notes like "V9 = suspicious
merchant" — that would be fabricated. Instead, this knowledge base describes
general fraud-analytics PATTERNS based on the *shape* of an anomaly: how many
features deviate, how strongly (z-scores), and interpretable fields like
Amount. The agent later builds a natural-language query describing a flagged
row's shape, and retrieves the most relevant pattern(s) here to reason with.

Uses ChromaDB with its built-in default embedding function (a small ONNX
model, downloaded automatically on first use) — no need for a separate
sentence-transformers/torch install, which is heavy and unnecessary here.
"""

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from typing import List, Dict

COLLECTION_NAME_ONLINE = "fraud_patterns"
COLLECTION_NAME_OFFLINE = "fraud_patterns_offline_tfidf"


class _TfidfEmbeddingFunction(EmbeddingFunction):
    """A fully offline embedding function using TF-IDF (no model download,
    no internet required). This exists as a fallback for network-restricted
    environments. On a normal machine with internet access, Chroma's default
    embedding function (a small MiniLM model) gives noticeably better semantic
    matching and is what you should use day to day — this is just a backup.
    """

    def __init__(self, corpus: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer()
        self._vectorizer.fit(corpus)

    def __call__(self, input: Documents) -> Embeddings:
        return self._vectorizer.transform(input).toarray().tolist()

# Each entry: a general, reusable pattern or business rule an analyst would
# actually apply when reviewing a flagged transaction. "id" and "category"
# are metadata for reference; "text" is what gets embedded and searched.
CASE_NOTES: List[Dict[str, str]] = [
    {
        "id": "pattern_card_testing",
        "category": "fraud_pattern",
        "text": (
            "Card testing pattern: multiple features show strong deviation "
            "(z-score above 3) while the transaction amount is very small, "
            "often under $5. This matches attackers validating stolen card "
            "numbers with tiny purchases before attempting a larger charge."
        ),
    },
    {
        "id": "pattern_single_large_anomaly",
        "category": "fraud_pattern",
        "text": (
            "Single large-deviation pattern: only one or two features deviate "
            "strongly, combined with an unusually high transaction amount. "
            "This is consistent with a one-off high-value fraudulent purchase "
            "rather than systematic testing."
        ),
    },
    {
        "id": "pattern_broad_mild_deviation",
        "category": "fraud_pattern",
        "text": (
            "Broad mild-deviation pattern: many features show small to moderate "
            "deviation (z-score 1.5 to 2.5) with no single dominant outlier. "
            "This is frequently a false positive caused by a legitimate but "
            "unusual purchase (e.g. holiday shopping, travel) rather than fraud."
        ),
    },
    {
        "id": "rule_review_threshold",
        "category": "business_rule",
        "text": (
            "Review threshold rule: any transaction with an anomaly score above "
            "0.6 should be routed to manual review before any automatic action "
            "is taken, regardless of amount."
        ),
    },
    {
        "id": "rule_low_amount_deprioritize",
        "category": "business_rule",
        "text": (
            "Low-amount deprioritization rule: flagged transactions under $2 are "
            "lower priority for immediate blocking, since the direct financial "
            "loss is small, but should still be logged as they may indicate the "
            "start of a card-testing sequence."
        ),
    },
    {
        "id": "pattern_repeat_flagging",
        "category": "fraud_pattern",
        "text": (
            "Repeat flagging pattern: if the same underlying account or card has "
            "been flagged more than once in a short window, escalate directly to "
            "a fraud analyst rather than routing through the standard review queue."
        ),
    },
    {
        "id": "rule_false_positive_seasonal",
        "category": "business_rule",
        "text": (
            "Seasonal false-positive rule: elevated anomaly rates during known "
            "high-spending periods (e.g. holidays) should be reviewed with a "
            "higher threshold, since normal spending behavior shifts and looks "
            "more unusual relative to the rest of the year."
        ),
    },
    {
        "id": "pattern_negative_deviation",
        "category": "fraud_pattern",
        "text": (
            "Negative-direction deviation pattern: features deviating strongly in "
            "the negative direction (large negative z-scores) rather than positive "
            "have historically shown weaker correlation with confirmed fraud cases "
            "and warrant a slightly lower urgency in triage."
        ),
    },
]


def build_knowledge_base(persist_directory: str = "./chroma_store", offline: bool = False) -> "chromadb.Collection":
    """Create (or load, if it already exists) a persistent Chroma collection
    populated with the case notes above.

    Using a PersistentClient means the embeddings are saved to disk — you
    don't need to re-embed every document each time you run the program.

    Args:
        offline: force the TF-IDF fallback embedding (no internet/model
            download needed). You normally don't need to set this — by
            default (offline=False) this function tries Chroma's real
            embedding model first (better semantic search, downloads
            automatically on first use) and only falls back to the offline
            TF-IDF mode automatically if that download fails.

            Online and offline modes are stored under different collection
            names on disk (they produce different embedding dimensions and
            can't share a collection), so switching between them is safe and
            never causes a dimension-mismatch error.
    """
    client = chromadb.PersistentClient(path=persist_directory)

    if offline:
        return _build_with_function(
            client, COLLECTION_NAME_OFFLINE, _TfidfEmbeddingFunction([note["text"] for note in CASE_NOTES])
        )

    try:
        # embedding_function omitted -> Chroma uses its default real embedding
        # model. Trigger it immediately with a throwaway query so a network
        # failure surfaces now, not later mid-use.
        collection = _build_with_function(client, COLLECTION_NAME_ONLINE, None)
        collection.query(query_texts=["connectivity check"], n_results=1)
        return collection
    except Exception as e:
        print(f"[knowledge_base] Online embedding model unavailable ({e}); falling back to offline TF-IDF mode.")
        return _build_with_function(
            client, COLLECTION_NAME_OFFLINE, _TfidfEmbeddingFunction([note["text"] for note in CASE_NOTES])
        )


def _build_with_function(client: "chromadb.PersistentClient", collection_name: str, embedding_function) -> "chromadb.Collection":
    """Get-or-create the collection with the given embedding function (or the
    Chroma default, if None), and populate it with CASE_NOTES if it's empty.
    """
    kwargs = {"name": collection_name}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    collection = client.get_or_create_collection(**kwargs)

    # Only add documents if the collection is currently empty, to avoid
    # duplicating entries every time this function runs.
    if collection.count() == 0:
        collection.add(
            ids=[note["id"] for note in CASE_NOTES],
            documents=[note["text"] for note in CASE_NOTES],
            metadatas=[{"category": note["category"]} for note in CASE_NOTES],
        )

    return collection


def retrieve(collection: "chromadb.Collection", query: str, top_k: int = 2) -> List[Dict]:
    """Retrieve the top_k most relevant case notes/rules for a given natural-language query.

    Args:
        collection: a Chroma collection returned by build_knowledge_base()
        query: natural-language description of the anomaly's shape (see
               agent.py in Week 3 for how this gets constructed from an AnomalyFlag)
        top_k: how many results to return
    """
    results = collection.query(query_texts=[query], n_results=top_k)

    # Chroma returns parallel lists (one per query); we only sent one query,
    # so we unpack index [0] from each.
    retrieved = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        retrieved.append({"text": doc, "category": meta["category"], "distance": dist})
    return retrieved


if __name__ == "__main__":
    import argparse

    # Command-line flag instead of a hardcoded value: by default this uses
    # Chroma's real embedding model (downloads automatically, needs internet).
    # Pass --offline only if you're on a network-restricted machine and need
    # the TF-IDF fallback instead.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline TF-IDF embeddings instead of downloading the real model (lower quality, no internet needed).",
    )
    args = parser.parse_args()

    # Quick manual test: build the KB, then try a couple of realistic queries
    # that mimic what the agent will generate from a flagged row.
    kb = build_knowledge_base(offline=args.offline)

    test_queries = [
        "3 features deviate strongly with z-scores above 3, transaction amount is $2.50",
        "1 feature deviates strongly, transaction amount is $850",
        "6 features show mild deviation around z-score 2, amount is $45",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        for r in retrieve(kb, q, top_k=2):
            print(f"   [{r['category']}] (distance={r['distance']:.3f}) {r['text'][:90]}...")
        print()
