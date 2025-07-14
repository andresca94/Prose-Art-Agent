# main_agent.py
# -*- coding: utf-8 -*-
"""
ProseGenerator Backend – runs on Judgeval 0.0.50 … latest GitHub
────────────────────────────────────────────────────────────────
• LangGraph + Pinecone RAG + cross-encoder rerank + few-shot style
• Iteratively calls itself until the prose hits the desired word-count window
• Optional GPT-Image-1 cover art
• Judgment Labs tracing, inline evaluations, custom metrics
"""

from __future__ import annotations

# ── std-lib / 3rd-party imports ───────────────────────────────
import os, uuid, time, base64, logging, mimetypes, requests, inspect, torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI

from pinecone import Pinecone
from sentence_transformers import CrossEncoder
import numpy as np

from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer, ClassifierScorer

try:
    # ≥ judgeval-py 0.0.53
    from judgeval.integrations.langgraph import (
        JudgevalCallbackHandler,
        add_evaluation_to_state,
    )
except ImportError:                       # very old builds (≤ 0.0.52)
    from judgeval.integrations.langgraph import JudgevalCallbackHandler

    # drop-in async fallback
    def add_evaluation_to_state(
        state, *, input, actual_output, scorers, model
    ):
        judgment.async_evaluate(
            input=input,
            actual_output=actual_output,
            scorers=scorers,
            model=model,
        )

# ── basic logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sigmoid = torch.nn.Sigmoid()

# ═════════════════════════════ ENV ═════════════════════════════
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

def _need(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise EnvironmentError(f"{var} is not set")
    return val

OAI_KEY  = _need("OPENAI_API_KEY")
PC_KEY   = _need("PINECONE_API_KEY")
JUDG_KEY = _need("JUDGMENT_API_KEY")
ORG_ID   = _need("JUDGMENT_ORG_ID")

STATIC_DIR, COVERS_DIR = ROOT / "static", ROOT / "static" / "covers"
COVERS_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════ CONSTANTS ═══════════════════════════
GPT_MODEL     = "gpt-4o"
EMBED_MODEL   = "text-embedding-3-large"
PC_INDEX      = "semantic-search-booksum-improved"
PC_ENV        = "us-east1-gcp"
NAMESPACE     = "default"
FEW_SHOT_TOPK = 2
SEARCH_TOPK   = 5
WC_TOLERANCE  = 100           # ± words
MAX_ATTEMPTS  = 5             # prose retries before giving up

# ═════════════════════════ CLIENTS ═════════════════════════════
judgment = Tracer(
    api_key=JUDG_KEY, organization_id=ORG_ID,
    project_name="prose-generator-agent", enable_monitoring=True,
)
if not hasattr(judgment, "record_metric"):            # very old builds
    judgment.record_metric = lambda *_, **__: None     # type: ignore

oa_client  = wrap(OpenAI(api_key=OAI_KEY))
chat_model = ChatOpenAI(model=GPT_MODEL, temperature=0.7, openai_api_key=OAI_KEY)

pc = Pinecone(api_key=PC_KEY, environment=PC_ENV)
try:
    index = pc.Index(PC_INDEX)
except Exception as e:
    logger.warning("Pinecone index unavailable – retrieval disabled: %s", e)
    index = None

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu"
)

# ═════════════════════ UNIVERSAL CLASSIFIER ════════════════════
_params = inspect.signature(ClassifierScorer).parameters
HAS_KEY     = "key"     in _params
HAS_SLUG    = "slug"    in _params
HAS_OPTIONS = "options" in _params

def safe_classifier(*, name: str, threshold: float,
                    conversation: list[dict], positive: str = "yes"):
    """Return a ClassifierScorer that works on any Judge-val version."""
    kw = dict(name=name, threshold=threshold, conversation=conversation)
    if HAS_SLUG: kw["slug"] = "binary"
    if HAS_KEY:  kw["key"]  = positive
    kw["options"] = {positive: 0.0, "no": 0.0}
    return ClassifierScorer(**kw)

# ══════════════════════ TRACED HELPERS ═════════════════════════
def _span(tag: str):
    return judgment.observe(span_type="tool", name=tag)

@_span("embed")
def embed(text: str) -> List[float]:
    return oa_client.embeddings.create(
        model=EMBED_MODEL, input=[text]
    ).data[0].embedding

@_span("pinecone_search")
def pinecone_search(q: str, k: int = 5):
    if not index:
        return []
    return index.query(
        vector=embed(q), top_k=k, namespace=NAMESPACE, include_metadata=True
    ).matches

@_span("rerank")
def rerank(q: str, matches: List[Dict]):
    if len(matches) < 2:
        return matches
    logits = cross_encoder.predict([[q, m.metadata.get("text", "")] for m in matches])
    order  = np.argsort(sigmoid(torch.tensor(logits)).numpy())[::-1]
    return [matches[i] for i in order]

def iterative_gpt(prompt: str, lo: int, hi: int):
    """Single-prompt call with up to three self-revisions."""
    for attempt in range(3):
        rsp = chat_model.invoke([HumanMessage(content=prompt)]).content
        wc  = len(rsp.split())
        if lo <= wc <= hi:
            return rsp.strip()
        time.sleep(attempt + 1)
        adj = "extend with more detail" if wc < lo else "condense length"
        prompt = f"Current draft:\n\n{rsp}\n\nPlease {adj}."
    return rsp.strip()

@_span("download_image")
def _download(url: str, dest: Path):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    dest.write_bytes(r.content)

# ═══════════════════════ STATE / GRAPH ═════════════════════════
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
    visual_prompt: str | None
    cover_prompt: str
    cover_path: str
    prose: str
    target_wc: int
    attempts: int
    pass_wc: bool

# ── graph nodes ────────────────────────────────────────────────
def create_overview(st: ProseState):
    beats = "\n".join(f"{i+1}. {b}" for i, b in enumerate(st["beats"]))
    sys = (
        "You are a synopsis writer.\n\n"
        f"Beats:\n{beats}\n\nSetting: {st['setting']}\nGenre: {st['genre']}\n"
        f"Style: {st['style']}\n\nWrite 2-3 concise paragraphs tying everything together."
    )
    st["overview"] = chat_model.invoke([HumanMessage(content=sys)]).content.strip()
    return st

def fetch_refs(st: ProseState):
    m = rerank(st["overview"], pinecone_search(st["overview"], SEARCH_TOPK))
    st["references"] = "\n\n".join(x.metadata.get("text", "") for x in m)
    return st

def few_shots(st: ProseState):
    m = pinecone_search("stylistic novel excerpt", FEW_SHOT_TOPK)
    st["few_shots"] = "\n\n".join(
        f"Excerpt:\n{x.metadata.get('text','')}\n---" for x in m
    )
    return st

def make_cover_prompt(st: ProseState):
    if not st["cover_style"]:
        return st
    style = st["cover_style"].replace("_", " ")
    syn   = st["overview"] or " ".join(st["beats"])[:500]
    hint  = f" User instruction: {st['visual_prompt']}" if st["visual_prompt"] else ""
    st["cover_prompt"] = f"{style}. Chapter illustration: {syn}.{hint}"
    return st

def gen_cover(st: ProseState):
    if not st["cover_prompt"]:
        return st
    dest = COVERS_DIR / f"{uuid.uuid4().hex}.png"
    img  = oa_client.images.generate(
        model="gpt-image-1", prompt=st["cover_prompt"], size="1024x1024", n=1
    ).data[0]
    if getattr(img, "url", None):          # usual path
        _download(img.url, dest)
    else:                                   # some orgs return only b64
        dest.write_bytes(base64.b64decode(img.b64_json))
    st["cover_path"] = str(dest)
    return st

def eval_cover(st: ProseState):
    if not st.get("cover_path"):
        return st
    p = Path(st["cover_path"])
    mime, _ = mimetypes.guess_type(p)
    durl = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    critique = oa_client.chat.completions.create(
        model=GPT_MODEL, max_tokens=60,
        messages=[
            {"role": "system", "content": "You are an art critic. ≤40 words; cite 3 details."},
            {"role": "user",
             "content": [{"type": "text", "text": st["cover_prompt"]},
                         {"type": "image_url", "image_url": {"url": durl}}]},
        ],
    ).choices[0].message.content.strip()

    add_evaluation_to_state(
        st, input=st["cover_prompt"], actual_output=critique,
        scorers=[AnswerRelevancyScorer(threshold=0.5)], model=GPT_MODEL,
    )
    if st.get("visual_prompt"):
        add_evaluation_to_state(
            st, input=st["visual_prompt"], actual_output=critique,
            scorers=[AnswerRelevancyScorer(threshold=0.5)], model=GPT_MODEL,
        )
    return st

def gen_prose(st: ProseState):
    beats = "\n".join(f"{i+1}. {b}" for i, b in enumerate(st["beats"]))
    chars = "".join(f"- {c['name']}: {c['description']}\n" for c in st["characters"])

    prompt = f"""
You are a skilled novelist. Write ≈{st['target_wc']} words.

Overview:
{st['overview']}

Beats:
{beats}

Characters:
{chars or 'None'}

Setting: {st['setting']}
Genre: {st['genre']}
Style: {st['style']}

Context (inspiration only – do NOT copy):
{st['references']}

Stylistic examples:
{st['few_shots']}
"""
    lo = max(100, st['target_wc'] - WC_TOLERANCE)
    hi = st['target_wc'] + WC_TOLERANCE
    st["prose"] = iterative_gpt(prompt, lo, hi)

    # ▸▸ keep the serialised state small for Judge-val
    st["references"] = st["references"][:2000]    # ≈2 KB is plenty
    st["few_shots"]  = ""                         # discard excerpts

    return st

def eval_prose(st: ProseState):
    """relevancy + word-count binary check + metrics / loop flag"""
    # 1 · relevancy to beats
    add_evaluation_to_state(
        st,
        input=" | ".join(st["beats"]),
        actual_output=st["prose"],
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        model=GPT_MODEL,
    )
    # 2 · word-count binary
    wc, tgt = len(st["prose"].split()), st["target_wc"]
    within  = abs(wc - tgt) <= WC_TOLERANCE
    add_evaluation_to_state(
        st,
        input=str(tgt),
        actual_output=("yes" if within else "no"),
        scorers=[safe_classifier(
            name="Word-count", threshold=0.1,
            conversation=[
                {"role": "system",
                 "content": "Return yes if draft length is within the range; otherwise no."},
                {"role": "user",
                 "content": f"Target {tgt}±{WC_TOLERANCE}, actual {wc}. yes/no"},
            ],
        )],
        model=GPT_MODEL,
    )
    # 3 · loop bookkeeping
    st["pass_wc"] = within or st["attempts"] >= MAX_ATTEMPTS - 1
    st["attempts"] += 1
    judgment.record_metric("word_count", wc)
    judgment.record_metric("within_range", int(within))
    judgment.record_metric("prose_attempts", st["attempts"])
    return st

# ═════════════════════ BUILD LangGraph ═════════════════════════
wf = StateGraph(ProseState)
wf.add_node("overview",      create_overview)
wf.add_node("refs",          fetch_refs)
wf.add_node("shots",         few_shots)
wf.add_node("cover_prompt",  make_cover_prompt)
wf.add_node("cover_gen",     gen_cover)
wf.add_node("cover_eval",    eval_cover)
wf.add_node("prose_gen",     gen_prose)
wf.add_node("prose_eval",    eval_prose)

wf.add_edge("overview", "refs")
wf.add_edge("refs",     "shots")
wf.add_edge("shots",    "cover_prompt")
wf.add_edge("cover_prompt", "cover_gen")
wf.add_edge("cover_gen", "cover_eval")
wf.add_edge("cover_eval", "prose_gen")
wf.add_edge("prose_gen",  "prose_eval")

def _wc_check(st: ProseState):
    return "good" if st["pass_wc"] else "retry"

wf.add_conditional_edges(
    "prose_eval", _wc_check, {"good": END, "retry": "prose_gen"}
)
wf.set_entry_point("overview")
graph = wf.compile()

# ═════════════════════ FASTAPI APP ════════════════════════════
app = FastAPI(title="ProseGenerator + JudgmentLabs")
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
    approx_word_count: int = 1500
    cover_style: Optional[str] = None
    visual_prompt: Optional[str] = None

class ProseResponse(BaseModel):
    prose_output: str
    cover_image: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.post("/generate-prose", response_model=ProseResponse)
async def generate(req: ProseRequest):
    handler = JudgevalCallbackHandler(judgment)
    state: ProseState = {
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
        "cover_prompt": "",
        "cover_path": "",
        "prose": "",
        "target_wc": req.approx_word_count,
        "attempts": 0,
        "pass_wc": False,
    }
    try:
        final = graph.invoke(state, config={"callbacks": [handler]})
    except Exception as e:
        logger.error("LangGraph run failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Backend error.")

    cover_url = (
        f"/static/covers/{Path(final['cover_path']).name}"
        if final.get("cover_path") else None
    )
    return {"prose_output": final["prose"], "cover_image": cover_url}

# ═════════════════════ DEV ENTRY-POINT ════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_agent:app", host="0.0.0.0", port=8000, reload=True)
