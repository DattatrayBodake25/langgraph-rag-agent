from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langsmith import traceable
from src.rag_pipeline import get_retriever
from src.reflection import reflect_answer
from src.utils import log

# PLAN NODE
@traceable(name="plan_node")
def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide if retrieval is needed based on query type.
    Smart heuristic: retrieval is skipped for greetings or meta-queries.
    """
    query = state.get("query", "").strip()
    log(f"[PLAN] Received Query: {query}")

    # Simple heuristic: if query is short or not factual → skip retrieval
    no_retrieve_patterns = ["hello", "hi", "who are you", "your name", "thank you"]
    if any(p in query.lower() for p in no_retrieve_patterns) or len(query.split()) < 3:
        state["retrieve_needed"] = False
        log("[PLAN] Retrieval skipped (non-factual or conversational query).")
    else:
        state["retrieve_needed"] = True
        log("[PLAN] Retrieval required (factual question detected).")

    return state

# RETRIEVE NODE
@traceable(name="retrieve_node")
def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant documents from ChromaDB."""
    if not state.get("retrieve_needed"):
        log("[RETRIEVE] Retrieval not required. Skipping...")
        state["retrieved_docs"] = []
        return state

    retriever = get_retriever(top_k=5)
    query = state.get("query", "")

    # Retrieve relevant documents (our retriever is a callable)
    try:
        docs = retriever(query)
    except Exception as e:
        log(f"[RETRIEVE] ERROR: {e}")
        docs = []

    if not docs:
        log("[RETRIEVE] No relevant context found.")
    else:
        log(f"[RETRIEVE] Retrieved {len(docs)} relevant documents.")

    # Store retrieved content
    state["retrieved_docs"] = [d.page_content for d in docs]
    return state

# ANSWER NODE
@traceable(name="answer_node")
def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer using LLM with retrieved context."""
    query = state.get("query", "")
    contexts = state.get("retrieved_docs", [])
    joined_context = "\n\n".join(contexts[:5]) if contexts else ""

    # Choose model for answering (separate from reflection)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if not joined_context:
        log("[ANSWER] No context found. Using zero-shot fallback.")
        prompt = (
            f"You are a helpful AI assistant. Answer the following question concisely and accurately.\n\n"
            f"Question:\n{query}\n\n"
            "If the question is unrelated to your knowledge or context, say 'I don’t have enough information.'"
        )
    else:
        prompt = (
            f"You are an expert assistant. Use the following context to answer accurately.\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Question:\n{query}\n\n"
            "If the context does not contain relevant information, say 'I don't have enough information based on the provided data.'"
        )

    log("[ANSWER] Generating response with LLM...")
    try:
        response = llm.invoke(prompt)
        answer_text = response.content.strip()
    except Exception as e:
        log(f"[ANSWER] ERROR: {e}")
        answer_text = "Sorry, an error occurred while generating the answer."

    log(f"[ANSWER] Done. Length: {len(answer_text)} chars.")
    state["answer"] = answer_text
    return state

# REFLECT NODE
@traceable(name="reflect_node")
def reflect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Reflect on answer relevance using separate reflection model."""
    log("[REFLECT] Running reflection phase...")
    reflection_state = reflect_answer(state)
    log("[REFLECT] Reflection completed.")
    return reflection_state

# BUILD AGENT GRAPH
def build_agent_graph() -> StateGraph:
    """Construct LangGraph-based agent workflow."""
    graph = StateGraph(dict)

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reflect", reflect_node)

    # Define flow
    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "reflect")
    graph.add_edge("reflect", END)

    log("[GRAPH] Agent graph successfully built and ready.")
    return graph