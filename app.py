from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_core import (
    RAGError,
    ask_alpha_cinema,
    load_index,
    recommend_movies,
    resolve_openai_generation_model,
)


load_dotenv(override=True)
ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = ROOT / "storage" / "alpha_cinema_index.json"
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "12"))
DEFAULT_GENERATION_MODEL = resolve_openai_generation_model(
    os.getenv("OPENAI_GENERATION_MODEL") or os.getenv("GENERATION_MODEL")
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    top_n_recommendations: int = Field(default=5, ge=1, le=10)
    generation_model: str = Field(default=DEFAULT_GENERATION_MODEL)
    session_id: str | None = Field(default=None, min_length=1, max_length=120)
    remember_history: bool = Field(default=True)
    chat_history: list["ChatMessage"] = Field(default_factory=list, max_length=20)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


AskRequest.model_rebuild()


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=10)


class ReloadResponse(BaseModel):
    status: str
    index_path: str
    chunk_count: int | None = None
    movie_count: int | None = None
    message: str


def resolve_index_path() -> Path:
    """Resolve đường dẫn index từ env, fallback về file mặc định trong storage."""
    index_path_value = os.getenv("INDEX_PATH")
    if index_path_value:
        index_path = Path(index_path_value)
        if not index_path.is_absolute():
            index_path = ROOT / index_path
        return index_path
    return DEFAULT_INDEX_PATH


def load_rag_index() -> tuple[dict | None, str | None]:
    """Load index đã build sẵn để API có thể phục vụ truy vấn ngay khi khởi động."""
    index_path = resolve_index_path()
    if not index_path.exists():
        return None, f"Index file not found: {index_path}. Run build_index.py first."
    try:
        return load_index(index_path), None
    except Exception as exc:
        return None, f"Failed to load index from {index_path}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo trạng thái dùng chung của FastAPI trong suốt vòng đời ứng dụng."""
    app.state.index_path = resolve_index_path()
    app.state.index_payload = None
    app.state.index_error = None
    app.state.chat_sessions = {}
    app.state.index_payload, app.state.index_error = load_rag_index()
    yield


app = FastAPI(
    title="Alpha Cinema RAG API",
    description="RAG backend for movie QA, recommendations, and policy support.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_ready_index() -> dict:
    """Trả về index đã nạp hoặc raise 503 nếu backend chưa sẵn sàng."""
    index_payload = getattr(app.state, "index_payload", None)
    index_error = getattr(app.state, "index_error", None)
    if index_payload is None:
        detail = index_error or "RAG index is not loaded."
        raise HTTPException(status_code=503, detail=detail)
    return index_payload


def trim_chat_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Giá»›i háº¡n sá»‘ message lÆ°u trong session Ä‘á»ƒ memory ngáº¯n háº¡n khÃ´ng phÃ¬nh ra."""
    return history[-MAX_SESSION_MESSAGES:]


def resolve_request_history(request: AskRequest) -> list[dict[str, str]]:
    """Æ¯u tiÃªn history do client gá»­i; náº¿u khÃ´ng cÃ³ thÃ¬ dÃ¹ng history Ä‘ang lÆ°u theo session."""
    explicit_history = [
        {"role": item.role, "content": item.content.strip()}
        for item in request.chat_history
        if item.content.strip()
    ]
    if explicit_history:
        return trim_chat_history(explicit_history)

    if not request.session_id:
        return []

    sessions = getattr(app.state, "chat_sessions", {})
    stored_history = sessions.get(request.session_id, [])
    return trim_chat_history(list(stored_history))


def persist_session_history(
    session_id: str | None,
    history: list[dict[str, str]],
    question: str,
    answer: str,
    enabled: bool = True,
) -> None:
    """LÆ°u láº¡i turn hiá»‡n táº¡i theo session_id Ä‘á»ƒ request sau cÃ³ thá»ƒ tiáº¿p máº¡ch há»™i thoáº¡i."""
    if not enabled or not session_id:
        return

    sessions = getattr(app.state, "chat_sessions", {})
    updated_history = trim_chat_history(
        history
        + [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    )
    sessions[session_id] = updated_history
    app.state.chat_sessions = sessions


@app.get("/")
def root() -> dict:
    """Health-check cơ bản để biết API sống và index đã được load hay chưa."""
    index_payload = getattr(app.state, "index_payload", None)
    return {
        "status": "ok",
        "message": "Alpha Cinema RAG API is running",
        "index_loaded": index_payload is not None,
        "movie_count": (index_payload or {}).get("movie_count"),
        "chunk_count": (index_payload or {}).get("chunk_count"),
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    """Endpoint QA chính: retrieve nguồn liên quan rồi sinh câu trả lời cho người dùng."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    index_payload = get_ready_index()
    history = resolve_request_history(request)
    generation_model = resolve_openai_generation_model(request.generation_model)
    try:
        result = ask_alpha_cinema(
            question=question,
            index_payload=index_payload,
            api_key=None,
            top_k=request.top_k,
            top_n_recommendations=request.top_n_recommendations,
            generation_model=generation_model,
            chat_history=history,
        )
        persist_session_history(
            session_id=request.session_id,
            history=history,
            question=question,
            answer=str(result.get("answer") or ""),
            enabled=request.remember_history,
        )
        if request.session_id:
            result["session_id"] = request.session_id
            result["history_message_count"] = len(
                getattr(app.state, "chat_sessions", {}).get(request.session_id, history)
            )
        return result
    except RAGError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str) -> dict:
    """XÃ³a memory táº¡m cá»§a má»™t phiÃªn chat Ä‘á»ƒ báº¯t Ä‘áº§u há»™i thoáº¡i má»›i."""
    sessions = getattr(app.state, "chat_sessions", {})
    existed = session_id in sessions
    sessions.pop(session_id, None)
    app.state.chat_sessions = sessions
    return {
        "status": "ok",
        "session_id": session_id,
        "cleared": existed,
    }


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict:
    """Endpoint gợi ý phim, chỉ chạy nhánh recommendation mà không sinh answer dạng QA."""
    index_payload = get_ready_index()
    try:
        recommendations = recommend_movies(
            request.query.strip(),
            index_payload=index_payload,
            api_key=None,
            top_n=request.top_n,
        )
    except RAGError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc

    return {
        "query": request.query,
        "recommendations": recommendations,
    }


@app.post("/reload", response_model=ReloadResponse)
def reload_index() -> ReloadResponse:
    """Nạp lại index từ file khi dữ liệu đã được build mới mà chưa muốn restart server."""
    index_payload, index_error = load_rag_index()
    app.state.index_payload = index_payload
    app.state.index_error = index_error
    if index_payload is None:
        raise HTTPException(status_code=503, detail=index_error or "Could not reload index")
    return ReloadResponse(
        status="ok",
        index_path=str(resolve_index_path()),
        chunk_count=index_payload.get("chunk_count"),
        movie_count=index_payload.get("movie_count"),
        message="Index reloaded successfully",
    )
