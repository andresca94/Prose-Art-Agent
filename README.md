# ProseGenerator — Full‑Stack AI Agent

LangGraph‑powered **AI writing assistant** that turns a short list of *beats* into a ≈1 500‑word passage **and** (optionally) illustrates the chapter with an AI‑generated cover.\
The agent self‑critiques its prose and image via Judge‑val.\
A lightweight React UI (boot‑strapped with **Create React App**, *not* Vite) provides a one‑click playground.

---

## 0 · Tech Stack & Key Libraries

| Layer                 | Library / Service                            | Purpose                                                        |
| --------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| **Agent graph**       | **LangGraph**                                | Declarative DAG & state‑machine orchestrating every step.      |
| LLM access            | **OpenAI Python SDK** + **LangChain‑OpenAI** | Chat, embeddings & GPT‑Image calls.                            |
| Vector store          | **Pinecone**                                 | RAG retrieval: Books‑um index → semantic context snippets.     |
| Re‑ranking            | **Sentence‑Transformers Cross‑Encoder**      | Reranks Pinecone hits for higher relevance.                    |
| Persistence / tracing | **Judge‑val Tracer**                         | Fine‑grained traces **plus** relevancy scoring (text & image). |
| API server            | **FastAPI** + **Uvicorn**                    | Exposes `/generate-prose` & static cover assets.               |
| Web UI                | **React 18** (Create React App)              | Simple form & result viewer (no Vite).                         |

Other helpers: `python‑dotenv`, `torch`, `requests`, `pillow`, etc.

> **GPU?** Optional. The conda env installs CPU wheels by default.

---

## 1 · Prerequisites

| Tool             | Version tested | Notes                          |
| ---------------- | -------------- | ------------------------------ |
| Python           | 3.12           | Managed via **conda**.         |
| conda / Mamba    | ≥ 23           | `conda --version`              |
| Git              | any            | Clone repo.                    |
| C/C++ tool‑chain | system default | Needed by a few Python wheels. |

---

## 2 · Clone the Repository

```bash
git clone https://github.com/andresca94/Prose-Art-Agent.git
cd Prose-Art-Agent
```

---

## 3 · Provision the Backend Environment

A pre‑built `environment.yml` is already committed at the repo root—no need to create it manually.

```bash
# from repo root
conda env create -f environment.yml
conda activate prose-art-gen
```

This pulls Python 3.12 plus every backend dependency (FastAPI, LangGraph, Pinecone, Judge‑val, etc.).
If you require a GPU build of **PyTorch**, install the appropriate CUDA/MPS wheel *before* running the commands above, then recreate the env.

---

## 4 · Project Layout · Project Layout

```
ProseGenerator/
├── backend/          # FastAPI + LangGraph agent
│   ├── main_agent.py
│   ├── static/
│   │   └── covers/   # AI images land here
│   └── .env          # ← you create this
├── frontend/         # CRA React app (no Vite)
│   ├── src/
│   └── package.json
└── environment.yml   # single source of dependency truth
```

`__pycache__` is ignored via `.gitignore`.\

---

## 5 · Secrets · `.env` (one‑time)

1. **Switch to the backend folder**  
   ```bash
   cd backend
   ```

2. **Create the file** (interactive *heredoc*; hit `ENTER`, paste your keys, then `CTRL‑D`):

   ```bash
   cat > .env <<'ENV'
   OPENAI_API_KEY=sk-________________________________________
   PINECONE_API_KEY=pc-______________________________________
   JUDGMENT_API_KEY=jv-______________________________________
   JUDGMENT_ORG_ID=org-_____________________________________
   ENV
## 6 · Run the Backend Server

```bash
# from repo root
conda activate prose-art-gen           # if not already
python backend/main_agent.py
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The server auto‑reloads on code edits (`uvicorn --reload` inside the script).

---

## 7 · Smoke‑Test the Endpoint

```bash
curl -X POST http://localhost:8000/generate-prose \
     -H "Content-Type: application/json" \
     -d '{
          "beats": [
            "Jack and Xander continue their excavation on the lunar surface.",
            "They discover an ancient alien artifact."
          ],
          "characters": [
            {"name": "Jack",   "description": "A veteran lunar miner"},
            {"name": "Xander", "description": "Jack's logical partner"}
          ],
          "setting": "A secluded lunar excavation site overshadowed by corporate exploitation",
          "genre":   "Hard Sci-Fi",
          "style":   "Dark, suspenseful atmosphere reminiscent of Alien",
          "cover_style": "oil_paint",
          "visual_prompt": "show the artifact pulsing with blue light"
        }' | jq
```

Expected response:

```jsonc
{
  "prose_output": "<≈1 500-word passage>",
  "cover_image": "/static/covers/abcd1234.png"
}
```

Open `http://localhost:8000/static/covers/abcd1234.png` to view the illustration.

---

## 8 · Frontend (React 18, CRA)


```bash
cd frontend
npm install         # installs React 18, axios, etc.
npm install @mui/material @mui/icons-material @emotion/react @emotion/styled # Install Material UI core and icon
npm start           # http://localhost:3000
```

The CRA dev‑server proxies API calls to `:8000` (see `package.json → proxy`).\
Feel free to migrate to Next.js or another framework, but remember to update this README.


---

Made with ☄️ **LangGraph** · ⚛️ **React** · 🔍 **Pinecone** · 🎨 **OpenAI GPT‑Image** · 🔬 **Judge‑val**.

