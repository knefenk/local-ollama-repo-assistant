
import os
import logging
from pathlib import Path
from tqdm import tqdm
from git import Repo, NULL_TREE


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document


# ----------------------------
# 1. CONFIGURATION
# ----------------------------
REPO_PATH = "C:/Users/knef/Desktop/Playground/agent/MBF_RL"     # e.g., "../myproject"
INDEX_DIR = "./data/repo_index"
OLLAMA_EMBED = "nomic-embed-text"
OLLAMA_LLM = "codellama:latest"

# ----------------------------
# 2. LOAD CODE FILES
# ----------------------------
def load_repo_code(repo_path):
    docs = []
    for file in Path(repo_path).rglob("*.*"):
        if file.suffix.lower() in [".py", ".js", ".ts", ".cpp", ".java", ".md", ".txt", ".json", ".yaml"]:
            try:
                text = file.read_text(errors="ignore")
                docs.append(Document(page_content=text, metadata={
                    "type": "code",
                    "path": str(file.relative_to(repo_path))
                }))
            except Exception:
                pass
    print(f"Loaded {len(docs)} code documents.")
    return docs

# ----------------------------
# 3. LOAD COMMITS + DIFFS
# ----------------------------
def load_git_commits(repo_path, max_commits=200):
    repo = Repo(repo_path)
    commits = []
    for commit in tqdm(list(repo.iter_commits("Dev", max_count=max_commits)), desc="Parsing commits"):
        message = commit.message.strip().replace("\n", " ")
        diff_text = ""
        try:
            diffs = commit.diff(commit.parents[0]) if commit.parents else commit.diff(NULL_TREE)
            for d in diffs:
                diff_text += f"File: {d.a_path or d.b_path}\n"
                if d.diff:
                    diff_text += d.diff.decode("utf-8", errors="ignore")[:1500] + "\n"
        except Exception:
            pass

        content = f"Commit: {commit.hexsha[:8]}\nDate: {commit.committed_datetime}\nMessage: {message}\nChanges:\n{diff_text}"
        commits.append(Document(page_content=content, metadata={
            "type": "commit",
            "hash": commit.hexsha,
            "author": commit.author.name,
            "date": str(commit.committed_datetime)
        }))
    print(f"Loaded {len(commits)} commits.")
    return commits

# ----------------------------
# 4. SPLIT & EMBED
# ----------------------------
def build_vectorstore(all_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(all_docs)
    print(f"Split into {len(split_docs)} chunks.")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED)
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=INDEX_DIR)
    print("Vector index built & saved.")
    return db

# ----------------------------
# 5. QUERY PIPELINE
# ----------------------------
def build_qa_chain(db):
    llm = OllamaLLM(model=OLLAMA_LLM)
    retriever = db.as_retriever(search_kwargs={"k": 6})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# ----------------------------
# 6. MAIN
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    os.makedirs(INDEX_DIR, exist_ok=True)
    logging.info("Loading code files...")
    code_docs = load_repo_code(REPO_PATH)
    logging.info("Loading git commits...")
    commit_docs = load_git_commits(REPO_PATH)
    all_docs = code_docs + commit_docs

    logging.info("Building vector store...")
    db = build_vectorstore(all_docs)
    logging.info("Building QA chain...")
    qa = build_qa_chain(db)

    logging.info("Ready! Ask something about your repo.")
    while True:
        logging.info("Waiting for user query...")
        q = input("🔍 Query> ").strip()
        if not q or q.lower() in ["exit", "quit"]:
            logging.info("Exiting interactive loop.")
            break
        logging.info(f"Processing query: {q}")
        try:
            result = qa.invoke({"query": q})
            logging.info("Query processed successfully.")
        except Exception as e:
            logging.error(f"Error during query processing: {e}")
            continue
        print("\n💡 Answer:\n", result["result"], "\n")
        print("📚 Sources:")
        for src in result["source_documents"]:
            print(" -", src.metadata.get("path") or src.metadata.get("hash"))
        print()
