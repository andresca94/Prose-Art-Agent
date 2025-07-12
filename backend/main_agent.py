# main_agent.py
# -*- coding: utf-8 -*-
"""
ProseGenerator Backend
───────────────────────────
• LangGraph agent for ~1500-word prose (RAG + re-rank + few-shot)  
• Optional cover-art generation & evaluation (user-selectable style)  
• Judgeval tracing + dual online evals (prose relevancy -and- cover visualization relevancy)  
"""

from __future__ import annotations
import os, uuid, logging, torch, base64, mimetypes, requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from pinecone import Pinecone
from sentence_transformers import CrossEncoder
import numpy as np

from judgeval.tracer import Tracer, wrap
from judgeval.integrations.langgraph import JudgevalCallbackHandler
from judgeval.scorers import AnswerRelevancyScorer


from openai import OpenAI



sigmoid = torch.nn.Sigmoid()


# ────────────────────────── ENV & DIRS ──────────────────────────
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

OAI_KEY  = os.getenv("OPENAI_API_KEY")
PC_KEY   = os.getenv("PINECONE_API_KEY")
JUDG_KEY = os.getenv("JUDGMENT_API_KEY")
ORG_ID   = os.getenv("JUDGMENT_ORG_ID")
for k, v in {"OPENAI_API_KEY": OAI_KEY, "PINECONE_API_KEY": PC_KEY,
             "JUDGMENT_API_KEY": JUDG_KEY, "JUDGMENT_ORG_ID": ORG_ID}.items():
    if not v:
        raise EnvironmentError(f"{k} not set")

STATIC_DIR   = ROOT / "static"
COVERS_DIR   = STATIC_DIR / "covers"
COVERS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── CONSTANTS ──────────────────────────
GPT_MODEL     = "gpt-4o-2024-08-06"
EMBED_MODEL   = "text-embedding-3-large"
PC_INDEX      = "semantic-search-booksum-improved"
PC_ENV        = "us-east1-gcp"
NAMESPACE     = "default"
DESIRED_MIN   = 1400
DESIRED_MAX   = 1600
FEW_SHOT_TOPK = 2
SEARCH_TOPK   = 5


# ───────────────────────────── CLIENTS ──────────────────────────
oa_client  = wrap(OpenAI(api_key=OAI_KEY))
chat_model = ChatOpenAI(model=GPT_MODEL, temperature=0.7, openai_api_key=OAI_KEY)

pc    = Pinecone(api_key=PC_KEY, environment=PC_ENV)
index = pc.Index(PC_INDEX)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

judgment = Tracer(api_key=JUDG_KEY, organization_id=ORG_ID,
                  project_name="prose-generator-agent", enable_monitoring=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────── HELPERS ────────────────────────────
def embed(text: str) -> List[float]:
    return oa_client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def pinecone_search(query: str, top_k: int = 5) -> List[Dict]:
    res = index.query(vector=embed(query), top_k=top_k,
                      namespace=NAMESPACE, include_metadata=True)
    return res.matches

def rerank(query: str, matches: List[Dict]) -> List[Dict]:
    if len(matches) < 2:
        return matches
    logits = cross_encoder.predict([[query, m.metadata.get("text", "")] for m in matches])
    scores = sigmoid(torch.tensor(logits)).numpy()
    order  = np.argsort(scores)[::-1]
    return [matches[i] for i in order]

def iterative_gpt(prompt: str) -> str:
    for _ in range(3):
        rsp = chat_model.invoke([HumanMessage(content=prompt)]).content
        wc  = len(rsp.split())
        if DESIRED_MIN <= wc <= DESIRED_MAX:
            return rsp.strip()
        adj = "extend with more detail" if wc < DESIRED_MIN else "condense length"
        prompt = f"Current draft:\n\n{rsp}\n\nPlease {adj}."
    return rsp.strip()

def _download(url: str, dest: Path):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    dest.write_bytes(r.content)

# ────────────────────────── LANGGRAPH STATE ─────────────────────
class ProseState(TypedDict):
    beats: List[str]
    characters: List[Dict]
    setting: str
    genre: str
    style: str
    overview: str
    references: str
    few_shots: str
    cover_style: str
    cover_prompt: str
    visual_prompt: str | None = None   # ← free-form user text
    cover_path: str
    prose: str
    _handler: JudgevalCallbackHandler

# ───────────────────────────── NODES ────────────────────────────
def create_overview(state: ProseState) -> ProseState:
    beats_blk = "\n".join(f"{i+1}. {b}" for i, b in enumerate(state["beats"]))
    prompt = (
        "You are a synopsis writer.\n\n"
        f"Beats:\n{beats_blk}\n\n"
        f"Setting: {state['setting']}\nGenre: {state['genre']}\nStyle: {state['style']}\n\n"
        "Write 2-3 concise paragraphs tying everything together."
    )
    state["overview"] = chat_model.invoke([HumanMessage(content=prompt)]).content.strip()
    return state

def fetch_references(state: ProseState) -> ProseState:
    matches = rerank(state["overview"], pinecone_search(state["overview"], SEARCH_TOPK))
    state["references"] = "\n\n".join(m.metadata.get("text", "") for m in matches)
    return state

def fetch_few_shots(state: ProseState) -> ProseState:
    matches = pinecone_search("stylistic novel excerpt", FEW_SHOT_TOPK)
    state["few_shots"] = "\n\n".join(f"Excerpt:\n{m.metadata.get('text','')}\n---" for m in matches)
    return state

# ───── cover-art nodes ─────
def create_cover_prompt(state: ProseState) -> ProseState:
    if not state["cover_style"]:
        return state
    style_tag = state["cover_style"].replace("_", " ")
    synopsis  = state["overview"] or " ".join(state["beats"])[:500]
    user_line = f" User instruction: {state['visual_prompt']}" if state["visual_prompt"] else ""
    state["cover_prompt"] = (
        f"{style_tag}. Chapter illustration for story: {synopsis}.{user_line}"
    )
    return state

def generate_cover_image(state: ProseState) -> ProseState:
    if not state["cover_prompt"]:
        return state

    fname = f"{uuid.uuid4().hex}.png"
    dest = COVERS_DIR / fname

    rsp = oa_client.images.generate(
        model="gpt-image-1",
        prompt=state["cover_prompt"],
        size="1024x1024",
        n=1,
    )

    img_data = rsp.data[0]

    try:
        if getattr(img_data, "url", None):
            _download(img_data.url, dest)
        elif getattr(img_data, "b64_json", None):
            dest.write_bytes(base64.b64decode(img_data.b64_json))
        else:
            raise ValueError("No valid image data returned from OpenAI")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise

    state["cover_path"] = str(dest)
    return state


def eval_cover(state: ProseState) -> ProseState:
    """
    Judge the relevance of the generated cover illustration to the prompt
    using Judge-val’s built-in AnswerRelevancyScorer and a real LLM judge.
    """
    if not state.get("cover_path"):
        return state

    # Build a data-URL from the saved image
    p = Path(state["cover_path"])
    mime, _ = mimetypes.guess_type(p)
    data_url = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"

    # Ask the judge model for a short critique (no numeric score needed)
    system_prompt = (
        "You are an art critic. In ONE short paragraph (<=40 words) "
        "state whether the illustration matches the prompt. "
        "Reference at least three concrete details from the prompt."
    )

    reply = (
        oa_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": state["cover_prompt"]},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=60,
        )
        .choices[0]
        .message
        .content
        .strip()
    )

    # Push the full critique to Judge-val for logging / relevancy scoring
    judgment.async_evaluate(
        input=state["cover_prompt"],
        actual_output=reply,
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        model="gpt-4o",
    )

    return state


# ───── prose generation nodes ─────
def generate_prose(state: ProseState) -> ProseState:
    beats_blk = "\n".join(f"{i+1}. {b}" for i, b in enumerate(state["beats"]))
    chars_blk = "".join(f"- {c['name']}: {c['description']}\n" for c in state["characters"])
    prompt = f"""
You are a skilled novelist. Write ≈1500 words of cohesive prose.

Overview:
{state['overview']}

Beats:
{beats_blk}

Characters:
{chars_blk or 'None'}

Setting: {state['setting']}
Genre: {state['genre']}
Style: {state['style']}

Context snippets (inspiration only, do **not** copy):
{state['references']}

Stylistic Examples:
{state['few_shots']}
"""
    state["prose"] = iterative_gpt(prompt)
    return state

def evaluate_prose(state: ProseState) -> ProseState:
    judgment.async_evaluate(
        input=" | ".join(state["beats"]),
        actual_output=state["prose"],
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        model=GPT_MODEL,
    )
    return state

# ──────────────────────────── GRAPH ────────────────────────────
wf = StateGraph(ProseState)
wf.add_node("create_overview",     create_overview)
wf.add_node("fetch_references",    fetch_references)
wf.add_node("fetch_few_shots",     fetch_few_shots)

wf.add_node("create_cover_prompt", create_cover_prompt)
wf.add_node("generate_cover_image",generate_cover_image)
wf.add_node("eval_cover",          eval_cover)

wf.add_node("generate_prose",      generate_prose)
wf.add_node("evaluate_prose",      evaluate_prose)

wf.add_edge("create_overview",     "fetch_references")
wf.add_edge("fetch_references",    "fetch_few_shots")
wf.add_edge("fetch_few_shots",     "create_cover_prompt")
wf.add_edge("create_cover_prompt", "generate_cover_image")
wf.add_edge("generate_cover_image","eval_cover")
wf.add_edge("eval_cover",          "generate_prose")
wf.add_edge("generate_prose",      "evaluate_prose")
wf.add_edge("evaluate_prose",      END)
wf.set_entry_point("create_overview")


agent_graph = wf.compile()

# ────────────────────────── FASTAPI ────────────────────────────
app = FastAPI(title="ProseGenerator RAG-Agent w/ Covers")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static/covers", StaticFiles(directory=COVERS_DIR), name="covers")

class ProseRequest(BaseModel):
    beats: List[str]
    characters: Optional[List[Dict]] = []
    setting: str = ""
    genre: str = ""
    style: str = ""
    cover_style: str | None = None
    visual_prompt: str | None = None      # key in COVER_STYLES

class ProseResponse(BaseModel):
    prose_output: str
    cover_image: Optional[str] = None  # URL

@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.post("/generate-prose", response_model=ProseResponse)
async def generate(req: ProseRequest):
    handler = JudgevalCallbackHandler(judgment)
    init_state: ProseState = {
        "beats": req.beats,
        "characters": req.characters or [],
        "setting": req.setting,
        "genre": req.genre,
        "style": req.style,
        "overview": "",
        "references": "",
        "few_shots": "",
        "cover_style": req.cover_style or "",
        "visual_prompt": req.visual_prompt or "",
        "cover_score": 0.0, 
        "cover_prompt": "",
        "cover_path": "",
        "prose": "",
        "_handler": handler,
    }
    final = agent_graph.invoke(init_state, config={"callbacks": [handler]})
    cover_url = (
        f"/static/covers/{Path(final['cover_path']).name}"
        if final.get("cover_path") else None
    )
    return {"prose_output": final["prose"], "cover_image": cover_url}

# ───────────────────────── DEV ENTRY ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_agent:app", host="0.0.0.0", port=8000, reload=True)
