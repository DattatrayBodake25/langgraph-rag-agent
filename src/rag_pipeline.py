import os
import re
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.utils import log

# Environment Setup
load_dotenv()

DATA_PATH = os.path.join(os.getcwd(), "data")
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.6  

# Text Cleaning Function
def clean_text(text: str) -> str:
    """Perform comprehensive text cleaning on a string."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\r\n\t]", " ", text)
    text = re.sub(r"[^\w\s.,;:!?()\-]", "", text)
    text = re.sub(r"([.,;:!?()\-])\1+", r"\1", text)
    text = text.strip()
    return text

# Document Loading
def load_documents(data_path: str):
    """Load .txt and .pdf documents from a folder and clean text."""
    docs = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()
        elif file_name.endswith(".pdf"):
            loader = UnstructuredPDFLoader(file_path)
            raw_docs = loader.load()
        else:
            continue

        for doc in raw_docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = file_name
        docs.extend(raw_docs)

    log(f"[INFO] Loaded and cleaned {len(docs)} documents from '{data_path}'.")
    return docs

# Document Splitting
def split_documents(docs):
    """Split documents into smaller chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n", ".", "!", "?"]
    )
    split_docs = splitter.split_documents(docs)
    log(f"[INFO] Split into {len(split_docs)} text chunks.")
    return split_docs


# Vector Store Creation
def create_vector_store(split_docs):
    """Create and persist Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    log(f"[INFO] Chroma vector store created and persisted at '{CHROMA_DB_DIR}'.")
    return vector_store

# Load or Create Vector Store
def load_or_create_vector_store():
    """Load existing Chroma vector store or create a new one."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        log("[INFO] Existing Chroma DB found. Loading...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
    else:
        log("[INFO] No existing Chroma DB found. Creating new vector store...")
        docs = load_documents(DATA_PATH)
        split_docs = split_documents(docs)
        vector_store = create_vector_store(split_docs)

    return vector_store


# Retriever with Similarity Filtering
def retrieve_with_filter(query: str, top_k: int = 5):
    """
    Retrieve relevant documents from Chroma with similarity filtering.
    Uses similarity_search_with_score() to filter by SIMILARITY_THRESHOLD.
    """
    vector_store = load_or_create_vector_store()

    try:
        results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        log(f"[RETRIEVE] ERROR during retrieval: {e}")
        return []

    if not results_with_scores:
        log("[RETRIEVE] No results returned from vector store.")
        return []

    filtered_results = []
    similarities = []

    for doc, score in results_with_scores:
        similarities.append(score)
        if score >= SIMILARITY_THRESHOLD:
            doc.metadata["score"] = score
            filtered_results.append(doc)

    log(f"[RETRIEVE] Retrieved {len(results_with_scores)} docs, "
        f"{len(filtered_results)} passed threshold ({SIMILARITY_THRESHOLD}).")

    for i, (doc, sim) in enumerate(zip([d for d, _ in results_with_scores], similarities), 1):
        log(f"   ↳ Doc {i} (score={sim:.3f}): {doc.metadata.get('source', 'unknown')}")

    return filtered_results

# Retriever Helper
def get_retriever(top_k: int = 3):
    """Return callable retriever for RAG queries."""
    def retriever_fn(query: str):
        docs = retrieve_with_filter(query, top_k=top_k)
        if not docs:
            log("[RETRIEVE] No relevant documents found above similarity threshold.")
        return docs
    return retriever_fn