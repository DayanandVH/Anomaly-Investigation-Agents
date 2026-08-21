"""
The LangGraph agent: takes a flagged anomaly, retrieves relevant fraud
patterns via RAG, and asks an LLM to produce a plain-English explanation and
a recommended action.

Graph shape (deliberately simple — two nodes in a line):

    START -> retrieve -> reason -> END

  retrieve: turns the anomaly's structured data (z-scores, amount) into a
            natural-language description of its "shape", and looks up the
            most relevant fraud patterns/business rules from the knowledge
            base (see knowledge_base.py).

  reason:   feeds the flagged features + retrieved patterns to the LLM and
            asks for a short explanation and a recommended action.

Why LangGraph and not just a plain function call chain? Two nodes is
overkill for LangGraph on its own, but structuring it this way sets up
naturally for later extensions (e.g. a third node that decides whether to
escalate, or a loop that re-queries the knowledge base if the first
retrieval wasn't useful) without restructuring everything.
"""

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from knowledge_base import retrieve as kb_retrieve


class AgentState(TypedDict):
    """The data that flows through the graph, growing at each node.
    TypedDict just gives us autocomplete/type-checking; at runtime it's a
    plain dict.
    """
    anomaly: Dict[str, Any]           # a flagged row's data (from AnomalyFlag, serialized to dict)
    query: str                        # natural-language description built from the anomaly's shape
    retrieved: List[Dict[str, str]]   # fraud patterns/rules pulled from the knowledge base
    explanation: str                  # the LLM's final explanation + recommended action


def describe_anomaly_shape(anomaly: Dict[str, Any]) -> str:
    """Turn a flagged row's structured data into a natural-language query
    describing its 'shape' (how many features deviate, how strongly, and the
    transaction amount) — this is what lets the knowledge base, which is
    built around general patterns, match this specific anomaly WITHOUT ever
    needing to know what an anonymized feature like 'V9' actually means.
    """
    n_features = len(anomaly["top_features"])
    max_z = max(abs(f["z_score"]) for f in anomaly["top_features"])
    amount = anomaly["raw_row"].get("Amount", "unknown")
    return (
        f"{n_features} features deviate with max absolute z-score {max_z:.2f}, "
        f"transaction amount is ${amount}"
    )


def build_agent(kb_collection, llm):
    """Assemble the LangGraph agent.

    Args:
        kb_collection: a Chroma collection from knowledge_base.build_knowledge_base()
        llm: any LangChain chat model with an .invoke() method — e.g.
             ChatOllama for real use, or a fake/stub model for testing the
             graph's wiring without needing a real LLM running.
    """

    def retrieve_node(state: AgentState) -> AgentState:
        """Node 1: build the query from the anomaly's shape, retrieve relevant patterns."""
        query = describe_anomaly_shape(state["anomaly"])
        results = kb_retrieve(kb_collection, query, top_k=2)
        return {**state, "query": query, "retrieved": results}

    def reason_node(state: AgentState) -> AgentState:
        """Node 2: ask the LLM to explain the anomaly and recommend an action,
        given the flagged features and the retrieved fraud patterns.
        """
        features_text = "\n".join(
            f"  - {f['feature']}: value={f['value']}, z-score={f['z_score']}"
            for f in state["anomaly"]["top_features"]
        )
        context_text = "\n".join(f"- {r['text']}" for r in state["retrieved"]) or "(no matching patterns found)"

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a fraud analyst assistant. Given a flagged transaction's "
             "feature deviations and relevant fraud patterns, produce: "
             "(1) a short plain-English explanation of why it was flagged, and "
             "(2) a recommended action. Be concise — 3 to 4 sentences total. "
             "Do not invent meanings for the anonymized feature names (e.g. V9) — "
             "reason only from their statistical deviation (z-score) and the "
             "provided patterns."),
            ("human",
             "Flagged transaction:\n"
             "Anomaly score: {score}\n"
             "Top deviating features:\n{features}\n\n"
             "Relevant fraud patterns / business rules:\n{context}\n\n"
             "Provide the explanation and recommended action."),
        ])

        chain = prompt | llm  # LangChain's pipe operator: run prompt, feed result into llm
        response = chain.invoke({
            "score": state["anomaly"]["anomaly_score"],
            "features": features_text,
            "context": context_text,
        })

        # Chat models return a message object; .content is the actual text.
        return {**state, "explanation": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", END)

    # .compile() turns the graph definition into something runnable (.invoke())
    return graph.compile()


if __name__ == "__main__":
    import sys
    from dataclasses import asdict

    from data import load_data
    from detector import AnomalyDetector
    from knowledge_base import build_knowledge_base

    use_offline_kb = "--offline" in sys.argv

    # --- Build the pieces from Weeks 1 and 2 ---
    df, source = load_data()
    feature_cols = [c for c in df.columns if c != "Class"]
    detector = AnomalyDetector(contamination=0.02).fit(df, feature_cols)
    flags = detector.score(df)
    print(f"Data source: {source} | {len(flags)} anomalies flagged\n")

    kb = build_knowledge_base(offline=use_offline_kb)

    # --- Choose the LLM ---
    if "--fake-llm" in sys.argv:
        # Stand-in LLM for testing the graph's wiring without a real model
        # running (used here in the sandbox, which can't reach Ollama).
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        llm = FakeListChatModel(responses=[
            "This transaction was flagged due to several features deviating well "
            "beyond normal ranges, consistent with a card-testing pattern. "
            "Recommended action: route to manual review and monitor for repeat "
            "small-amount transactions on the same account."
        ])
    elif "--groq" in sys.argv:
        # Free-tier hosted API fallback — use this if local Ollama isn't working
        # on your machine (e.g. GPU/driver incompatibility). Requires a free API
        # key from https://console.groq.com/keys, set as an environment variable:
        #   export GROQ_API_KEY="your-key-here"
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    else:
        # Real use: talk to a locally running Ollama server.
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.1:8b", temperature=0)

    agent = build_agent(kb, llm)

    # --- Run the agent on the first couple of flagged anomalies ---
    for flag in flags[:2]:
        anomaly_dict = asdict(flag)
        result = agent.invoke({"anomaly": anomaly_dict, "query": "", "retrieved": [], "explanation": ""})

        print(f"--- Row {flag.row_index} (anomaly_score={flag.anomaly_score}) ---")
        print(f"Query used for retrieval: {result['query']}")
        print(f"Retrieved patterns: {[r['category'] for r in result['retrieved']]}")
        print(f"\nAgent explanation:\n{result['explanation']}\n")
