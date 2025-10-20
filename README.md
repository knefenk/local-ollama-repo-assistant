# 🧠 Local Code Q&A Assistant (Ollama + RAG)

A lightweight local assistant that indexes your codebase and Git commit history, then answers natural-language questions about it using locally-hosted Ollama models.  
No API keys, no external calls — runs fully offline.

---

## 🚀 Features
- Indexes code files and recent Git commits  
- Builds a local Chroma vector index with embeddings  
- Uses Ollama models for Retrieval-Augmented Generation (RAG)  
- Supports both local and cloned repositories  
- Clean CLI interface for quick Q&A  

---

## 🧩 Requirements
- Python 3.10+  
- Ollama installed and running locally  
- Git installed  
- (Optional) GPU with ≥8 GB VRAM for faster inference  

---

## ⚙️ Setup

### 1. Clone this project
```bash
git clone <this-repo-url>
cd repo_rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# or
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
ollama pull codellama:latest
ollama pull nomic-embed-text
```

---

## 🧱 Repository Options

### Option A: Use an existing local repo
Edit the config in `main.py`:

```python
REPO_PATH = "C:/path/to/your/local/repo"
```

### Option B: Clone a GitHub repo to analyze
You can clone any public repo before indexing:

```bash
git clone https://github.com/username/project.git ./target_repo
```

Then set:

```python
REPO_PATH = "./target_repo"
```

---

## ▶️ Run

```bash
python main.py
```

Example session:

```
🔍 Query> What changes were made to api_handler.py recently?
💡 Answer: ...
📚 Sources:
 - api_handler.py
 - commit 3f9a2b7c
```

---

## 💻 Recommended Hardware

| Setup | CPU | RAM | GPU | Models |
|-------|-----|-----|-----|--------|
| Basic (CPU only) | 6–8 cores | 16–32 GB | None | `phi3`, `mistral` |
| Developer | 8 cores | 32 GB | ≥8 GB VRAM | `codellama:7b` |
| Advanced | 16 cores | 64 GB | ≥16 GB VRAM | `codellama`, `llama3` |

---

## 🪄 Notes
- The first run indexes all code and commits into `./data/repo_index`
- To rebuild the index, delete that folder and rerun
- You can change the target branch in `load_git_commits()` (default `"Dev"`)
- Works entirely offline once models are downloaded

---

## 🧭 Next Steps
- Add a reflection or summarization layer (e.g., `phi3` as secondary model)
- Integrate MCP for IDE-style interactions
- Build a simple web or VS Code interface

---

**Made for local development and full data privacy.**

