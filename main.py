import logging
from pathlib import Path
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


# CONFIG
REPO_PATH = "./okvis2"
INDEX_DIR = "./data/okvis2"
OLLAMA_EMBED = "nomic-embed-text"
OLLAMA_LLM = "codellama:latest"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_code_docs(repo_path: str):
    docs = []
    p = Path(repo_path)
    exts = {".py", ".js", ".ts", ".cpp", ".java", ".md", ".txt", ".json", ".yaml", ".yml"}
    for f in p.rglob("*.*"):
        if f.suffix.lower() in exts:
            try:
                text = f.read_text(errors="ignore")
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"path": str(f.relative_to(p)), "extension": f.suffix}))
            except Exception as e:
                logging.warning(f"read fail {f}: {e}")
    logging.info(f"Loaded {len(docs)} files")
    return docs

def build_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    logging.info("Splitting documents...")
    split_docs = splitter.split_documents(docs)
    logging.info(f"{len(split_docs)} chunks")

    logging.info("Creating embeddings and Chroma index...")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED)
    db = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=INDEX_DIR)
    logging.info("Index ready")
    return db, split_docs

def build_bm25(split_docs):
    logging.info("Building BM25 index...")
    bm25 = BM25Retriever.from_documents(split_docs)
    bm25.k = 6
    return bm25

def hybrid_search(query: str, db, bm25, k=6):
    # vector retrieval
    vect_retriever = db.as_retriever(search_kwargs={"k": k})
    vdocs = vect_retriever._get_relevant_documents(query, run_manager=CallbackManagerForRetrieverRun.get_noop_manager())
    bdocs = bm25._get_relevant_documents(query, run_manager=CallbackManagerForRetrieverRun.get_noop_manager())
    # merge keeping order and dedupe by content
    seen = set()
    merged = []
    for d in (vdocs + bdocs):
        key = (d.metadata.get("path"), d.page_content[:200])
        if key not in seen:
            seen.add(key)
            merged.append(d)
            if len(merged) >= k:
                break
    return merged

def generate_answer(prompt: str, model_name=OLLAMA_LLM):
    llm = OllamaLLM(model=model_name)
    return llm.invoke(prompt)

if __name__ == "__main__":
    docs = load_code_docs(REPO_PATH)
    db, split_docs = build_index(docs)
    bm25 = build_bm25(split_docs)

    print("Ready. Type queries (exit to quit).")
    while True:
        q = input("> ").strip()
        if not q or q.lower() in ("exit","quit"):
            break
        hits = hybrid_search(q, db, bm25, k=6)
        context = "\n\n".join([h.page_content[:1000] for h in hits])
        prompt = f"Context:\n{context}\n\nQuestion:\n{q}\n\nAnswer concisely:"
        ans = generate_answer(prompt)
        print("\n--- ANSWER ---\n", ans, "\n\n--- SOURCES ---")
        for i,h in enumerate(hits,1):
            print(f"[{i}] {h.metadata.get('path')} ({h.metadata.get('extension')})")
