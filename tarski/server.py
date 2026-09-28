"""Local HTTP API.

  POST /v1/decide     {"text" or "texts", "tasks": [...]}  -> label, probabilities and confidence per task
  POST /v1/systemone  Jev's wire format. A question is answered by the trained branch whose task name is
                      the question id (or the question's optional "task" field). Its options must be the
                      branch's labels, or a subset of them (probabilities are renormalised over the subset).
                      Questions with no branch, and optionally answers below a confidence threshold, go to
                      a fallback Jev-compatible server (laya, OpenJev, stuntd, Jev) when one is configured.
  GET  /v1/tasks      trained tasks, their size, split depth, test metrics and whether they are resident
  POST /v1/tasks/{task}/load | /unload

Clients built for Jev (typesafe-sdk, OpenJev routes, stuntd) work against /v1/systemone unchanged.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from tarski.engine import Engine, confidence

MODEL_NAME = "tarski-local"
TRUE_WORDS = ("true", "yes", "1", "y", "positive")


class DecideRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    tasks: List[str] = Field(min_length=1)


class SystemOneRequest(BaseModel):
    state: Any
    model: str = MODEL_NAME
    questions: Dict[str, Dict[str, Any]] = Field(min_length=1)


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _positive_index(labels: List[str]) -> int:
    for i, l in enumerate(labels):
        if l.strip().lower() in TRUE_WORDS:
            return i
    return len(labels) - 1


def jev_answer(q: Dict, labels: List[str], p: np.ndarray) -> Dict:
    """Shape a branch's distribution as Jev answers the question type asked."""
    kind = q.get("type")
    if kind == "noul":
        if len(labels) != 2:
            raise ValueError(f"a noul question needs a two-label branch, this one has {len(labels)} labels")
        return {"type": "noul", "noul": float(p[_positive_index(labels)])}
    if kind == "choice":
        opts = list(q.get("criteria") or labels)
        missing = [o for o in opts if o not in labels]
        if missing:
            raise ValueError(f"options {missing} are not labels of this branch ({labels})")
        sub = np.array([p[labels.index(o)] for o in opts], dtype=np.float64)
        sub = sub / sub.sum() if sub.sum() > 0 else np.full(len(opts), 1 / len(opts))
        top = int(sub.argmax())
        return {"type": "choice", "choice": opts[top], "probabilities": {o: float(v) for o, v in zip(opts, sub)},
                "confidence": confidence(sub)}
    if kind == "score":
        levels = list(q.get("criteria") or labels)
        if len(levels) != len(labels):
            raise ValueError(f"score has {len(levels)} levels, the branch has {len(labels)} labels")
        order = sorted(range(len(labels)), key=lambda i: int(labels[i]) if labels[i].isdigit() else i)
        pp = np.array([p[i] for i in order], dtype=np.float64)
        return {"type": "score", "score": float(sum(i * v for i, v in enumerate(pp))),
                "legend": {str(i): levels[i] for i in range(len(levels))},
                "probabilities": {str(i): float(v) for i, v in enumerate(pp)}, "confidence": confidence(pp)}
    raise ValueError(f"unknown question type {kind!r}")


def create_app(engine: Engine, fallback_url: Optional[str] = None, fallback_model: str = "jev-latest",
               fallback_key: Optional[str] = None, escalate_below: Optional[float] = None,
               api_key: Optional[str] = None, escalate_oos: bool = False) -> FastAPI:
    """`escalate_oos`: send questions whose message the branch flags as out of scope to the fallback."""
    app = FastAPI(title="tarski", version="0.2")

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1/"):
            if request.headers.get("authorization", "") != f"Bearer {api_key}":
                return JSONResponse({"detail": {"error_type": "authentication_error",
                                                "message": "invalid API key"}}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"ok": True, "base": engine.trunk.base, "device": str(engine.trunk.device)}

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": MODEL_NAME, "description": f"trained branches on {engine.trunk.base}"}]}

    @app.get("/v1/tasks")
    def tasks():
        out = {}
        for name, m in engine.tasks().items():
            out[name] = {k: m.get(k) for k in ("kind", "split", "depth", "labels", "params", "bytes", "loaded",
                                                "dataset", "metrics", "oos")}
        return {"tasks": out, "stats": {k: v for k, v in engine.stats.items() if k != "load_ms"}}

    @app.post("/v1/tasks/{task}/load")
    def load(task: str):
        t0 = time.perf_counter()
        try:
            engine.branch(task)
        except KeyError:
            raise HTTPException(404, f"no trained task {task!r}")
        return {"task": task, "load_ms": (time.perf_counter() - t0) * 1000}

    @app.post("/v1/tasks/{task}/unload")
    def unload(task: str):
        engine.unload(task)
        return {"task": task, "loaded": False}

    @app.post("/v1/decide")
    def decide(req: DecideRequest):
        texts = req.texts if req.texts is not None else ([req.text] if req.text is not None else None)
        if not texts:
            raise HTTPException(422, "give text or texts")
        known = engine.tasks()
        missing = [t for t in req.tasks if t not in known]
        if missing:
            raise HTTPException(404, f"no trained task(s) {missing}; trained: {sorted(known)}")
        r = engine.decide(texts, req.tasks)
        out = []
        for probs, oos in zip(r["results"], r["oos"]):
            row = {}
            for t, p in probs.items():
                labels = engine.labels(t)
                row[t] = {"label": labels[int(p.argmax())], "confidence": confidence(p),
                          "probabilities": {l: float(v) for l, v in zip(labels, p)}}
                if t in oos:
                    row[t]["out_of_scope"], row[t]["oos_score"] = oos[t]["flag"], oos[t]["score"]
            out.append(row)
        return {"results": out, "timing_ms": r["timing_ms"], "input_tokens": r["input_tokens"]}

    def ask_fallback(state: Any, questions: Dict[str, Dict]) -> Dict:
        import httpx

        headers = {"Authorization": f"Bearer {fallback_key}"} if fallback_key else {}
        body = {"state": state, "model": fallback_model, "questions": questions}
        resp = httpx.post(fallback_url.rstrip("/") + "/v1/systemone", json=body, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(502, f"fallback {fallback_url} answered {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        known = engine.tasks()
        local, remote = {}, {}
        for qid, q in req.questions.items():
            task = q.get("task", qid)
            (local if task in known else remote)[qid] = q
        if remote and not fallback_url:
            raise HTTPException(400, f"no trained branch for question(s) {sorted(remote)} and no fallback "
                                     f"configured; trained tasks: {sorted(known)}")
        answers, sources, usage, flagged = {}, {}, 0, []
        if local:
            tasks = [q.get("task", qid) for qid, q in local.items()]
            r = engine.decide([state_text(req.state)], tasks)
            usage += r["input_tokens"]
            for (qid, q), task in zip(local.items(), tasks):
                try:
                    a = jev_answer(q, engine.labels(task), r["results"][0][task])
                except ValueError as e:
                    raise HTTPException(400, f"question {qid!r}: {e}")
                conf = a.get("confidence", max(a.get("noul", 0.5), 1 - a.get("noul", 0.5)) * 2 - 1)
                oos = r["oos"][0].get(task, {}).get("flag", False)
                if oos:
                    flagged.append(qid)
                if fallback_url and ((escalate_below is not None and conf < escalate_below) or (escalate_oos and oos)):
                    remote[qid] = q
                else:
                    answers[qid], sources[qid] = a, "local"
        if remote:
            fb = ask_fallback(req.state, {qid: {k: v for k, v in q.items() if k != "task"} for qid, q in remote.items()})
            for qid in remote:
                answers[qid], sources[qid] = fb["answers"][qid], "fallback"
            usage += int(fb.get("usage", {}).get("input_tokens", 0))
        ordered = {qid: answers[qid] for qid in req.questions}
        return JSONResponse({"model": MODEL_NAME, "answers": ordered,
                             "usage": {"input_tokens": usage, "output_tokens": 0}},
                            headers={"X-Tarski-Sources": ",".join(f"{q}={s}" for q, s in sources.items()),
                                     "X-Tarski-Out-Of-Scope": ",".join(flagged)})

    return app
