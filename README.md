A production-grade Retrieval-Augmented Generation (RAG) system built to analyze Rwandan agricultural strategy documents, livestock plans, and policy frameworks.

## Architecture

- **Backend Framework:** FastAPI with Streaming Responses
- **Frontend UI:** Streamlit
- **Vector Database:** ChromaDB (Persisted locally)
- **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`)
- **LLM Engine:** ChatGroq (Dynamic model selector prioritizing Llama-3.3-70b / Llama-3.1-8b)
- **Retrieval Strategy:** Maximal Marginal Relevance (MMR) (`k=5`, `fetch_k=20`, `lambda_mult=0.7`)

## Key Features

1. **Diverse Document Retrieval:** Uses MMR to prevent duplicate information from being sent to the LLM context.
2. **Strict Factual Alignment:** System prompts restrict responses strictly to ingested MINAGRI context.
3. **Database Audit Script:** Independent ChromaDB inspection script (`check_db.py`) to verify collection counts and file registries.
4. **SQLite Compatibility Patch:** Embedded `db_patch.py` module to handle Python environment SQLite requirements smoothly.

## Dataset & Metrics

- **Indexed Chunks:** 7,379 chunks
- **Ingested Policy PDFs:** 20 complete strategic documents (PSTA 4, PSTA 5, Livestock Strategy, Irrigation Master Plan, etc.)

## Getting Started

### 1. Installation
```bash
git clone [https://github.com/thelmaBangsi/Production_rag_system.git](https://github.com/thelmaBangsi/Production_rag_system.git)
cd Production_rag_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Configure Environment
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
ANONYMIZED_TELEMETRY=False
3. Running the Application
Start the FastAPI backend:

Bash
uvicorn app:app --reload --port 8000
In a separate terminal tab, launch the Streamlit frontend:

Bash
streamlit run ui.py
4. Database Audit
To verify total document chunks and indexed PDF sources:

Bash
python check_db.py
EOF


---

### Step 2: Stage, Commit, and Push to GitHub

Once the file is created, commit and push it to your repository:

```bash
git add README.md
git commit -m "docs: add project documentation and setup guide"
git push
