import sys
from src.graph_builder import build_agent_graph
from src.utils import log

def run_agent(query: str):
    """
    Run the AI agent workflow for a given user query.
    This executes the LangGraph agent (plan → retrieve → answer → reflect).
    """
    try:
        log("Building agent workflow...", "INFO")
        agent_graph = build_agent_graph().compile()

        log("Running agent workflow...", "INFO")
        result = agent_graph.invoke({"query": query})

        # Retrieve results safely
        answer = result.get("answer", "No answer generated.")
        reflection_score = result.get("reflection_score", "N/A")
        retrieved_docs = result.get("retrieved_docs", [])

        # Display results to console
        log(f"Final Answer:\n{answer}", "SUCCESS")
        log(f"Reflection Score: {reflection_score}", "INFO")
        log(f"Retrieved {len(retrieved_docs)} document chunks.", "INFO")

        return result

    except Exception as e:
        log(f"Error occurred while running agent: {str(e)}", "ERROR")
        return {"error": str(e)}

if __name__ == "__main__":
    # Handle both CLI arguments and interactive input
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your question: ").strip()

    if not user_query:
        log("Please provide a valid question.", "WARNING")
        sys.exit(0)

    log(f"User Query: {user_query}", "INFO")
    result = run_agent(user_query)

    if "error" in result:
        log("Agent run failed. Please check the logs above.", "ERROR")