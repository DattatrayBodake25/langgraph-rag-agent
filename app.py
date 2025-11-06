import streamlit as st
from src.main import run_agent
from src.utils import log

# Streamlit Page Configuration
st.set_page_config(
    page_title="Generative AI RAG Agent",
    layout="centered",
    page_icon="🤖"
)

# App Header
st.title("Generative AI RAG Agent")
st.markdown(
    """
    This app uses **LangGraph**, **LangChain**, and **RAG (Retrieval-Augmented Generation)**  
    to answer questions from your **local knowledge base**.
    """
)
st.divider()

# User Input Section
user_query = st.text_area(
    "Enter your question below:",
    placeholder="e.g., What are the benefits of renewable energy?",
    height=120
)


# Run Agent Button
if st.button("Run Agent"):
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Running agent workflow... please wait."):
            result = run_agent(user_query)


        # Display Results
        st.success(" Agent workflow completed successfully!")

        # Final Answer
        st.subheader(" Final Answer")
        st.write(result.get("answer", " No answer generated."))


        # Reflection Score
        reflection_score = result.get("reflection_score", "N/A")
        try:
            reflection_value = float(reflection_score)
        except ValueError:
            reflection_value = None

        st.metric(
            label="Relevance & Completeness Score",
            value=reflection_score,
            delta=None if not reflection_value else (
                "High" if reflection_value > 0.7 else "Low"
            )
        )


        # Retrieved Contexts
        with st.expander(" View Retrieved Contexts"):
            retrieved_docs = result.get("retrieved_docs", [])
            if not retrieved_docs:
                st.info("No relevant documents were retrieved from the knowledge base.")
            else:
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Document {i}:**")
                    st.write(doc[:1200] + ("..." if len(doc) > 1200 else ""))
                    st.divider()


# Footer
st.divider()
st.markdown(
    """
    **Tech Stack:** LangGraph · LangChain · ChromaDB · Hugging Face · GPT-4o-mini · LangSmith · Streamlit  
    **Developer:** Dattatray Bodake | *Internship Test – SwarmLens 2025*  
    """
)