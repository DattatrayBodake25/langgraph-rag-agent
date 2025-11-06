from langchain_openai import ChatOpenAI
from typing import Dict, Any
from src.utils import log

def reflect_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflect on the generated answer for relevance and completeness.
    Uses a lightweight LLM to judge the answer and provide a numeric score.
    """

    query = state.get("query", "")
    answer = state.get("answer", "")

    if not query or not answer:
        log("[REFLECT] Missing query or answer. Skipping reflection.")
        state["reflection_score"] = None
        state["reflection"] = "No reflection — incomplete input."
        return state

    reflection_prompt = (
        "You are an AI evaluator. Rate how relevant and complete the following answer is "
        "in addressing the given question. Provide:\n"
        "- A relevance score between 0 and 1 (1 = perfect relevance)\n"
        "- A one-sentence reasoning.\n\n"
        f"Question: {query}\n"
        f"Answer: {answer}\n\n"
        "Respond in JSON format, for example: {\"score\": 0.85, \"reason\": \"Answer covers main points accurately.\"}"
    )

    # Use a smaller / cheaper model for reflection to separate from main answer model
    llm_reflector = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    log("[REFLECT] Evaluating answer relevance...")
    try:
        response = llm_reflector.invoke(reflection_prompt)
        raw_reflection = response.content.strip()
    except Exception as e:
        log(f"[REFLECT] ERROR: {e}")
        state["reflection_score"] = None
        state["reflection"] = "Error during reflection."
        return state

    # Parse score safely
    import re, json

    reflection_score = None
    reason = "Could not parse reflection output."

    try:
        # Attempt JSON parsing first
        parsed = json.loads(raw_reflection)
        reflection_score = float(parsed.get("score", 0))
        reason = parsed.get("reason", "").strip() or reason
    except Exception:
        # Fallback: extract first number between 0 and 1 using regex
        match = re.search(r"0\.\d+|1(\.0+)?", raw_reflection)
        if match:
            reflection_score = float(match.group())
            reason = raw_reflection
        else:
            reflection_score = None
            reason = raw_reflection

    # Normalize score range
    if reflection_score is not None:
        reflection_score = max(0.0, min(1.0, reflection_score))

    log(f"[REFLECT] Relevance Score: {reflection_score} | Reason: {reason}")
    state["reflection_score"] = reflection_score
    state["reflection"] = reason

    # Optional flag for evaluation
    if reflection_score is not None and reflection_score >= 0.7:
        state["is_relevant"] = True
        log("[REFLECT] ✅ Answer deemed relevant and sufficient.")
    else:
        state["is_relevant"] = False
        log("[REFLECT] ⚠️ Answer may be incomplete or off-topic.")

    return state