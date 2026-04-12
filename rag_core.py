from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

import numpy as np

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:  # pragma: no cover - optional until user installs deps
    firebase_admin = None
    credentials = None
    firestore = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:  # pragma: no cover - optional until user installs deps
    genai = None
    types = None
    ClientError = Exception

try:
    from openai import APIError as OpenAIAPIError
    from openai import OpenAI, RateLimitError as OpenAIRateLimitError
except Exception:  # pragma: no cover - optional until user installs deps
    OpenAI = None
    OpenAIAPIError = Exception
    OpenAIRateLimitError = Exception

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional until user installs deps
    PdfReader = None


SUPPORTED_TEXT_EXTENSIONS = frozenset({".md", ".txt", ".pdf", ".docx"})
DOCX_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
INTENT_LABELS = ("policy", "movie", "recommendation", "mixed")
BM25_K1 = 1.5
BM25_B = 0.75


class RAGError(RuntimeError):
    """Raised when the RAG pipeline cannot proceed."""


@dataclass
class Chunk:
    chunk_id: str
    source: str
    domain: str
    title: str
    section: str
    heading_path: list[str]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    search_text: str = ""
    term_freq: dict[str, int] = field(default_factory=dict)
    doc_len: int = 0


@dataclass
class MovieDocument:
    movie_id: str
    title: str
    origin_name: str
    slug: str
    year: int | None
    movie_type: str
    status: str
    age_rating: str
    categories: list[str]
    countries: list[str]
    actors: list[str]
    directors: list[str]
    is_kids_friendly: bool
    content: str
    search_keywords: list[str]
    poster_url: str
    thumb_url: str
    modified_time: str
    search_text: str = ""
    term_freq: dict[str, int] = field(default_factory=dict)
    doc_len: int = 0
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class SourceDocument:
    path: str
    file_type: str
    text: str
    hash: str
    size_bytes: int
    characters: int


@dataclass
class RetrievalResult:
    rank: int
    score: float
    sparse_score: float
    dense_score: float
    metadata_boost: float
    chunk_id: str
    source: str
    domain: str
    title: str
    section: str
    heading_path: list[str]
    content: str
    metadata: dict[str, Any]


def sha256_text(text: str) -> str:
    """Tạo fingerprint ổn định để theo dõi thay đổi của tài liệu và dữ liệu phim."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dumps(data: Any) -> str:
    """Serialize JSON có indent để file index dễ đọc khi debug."""
    return json.dumps(data, ensure_ascii=False, indent=2)


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng để giảm nhiễu trước khi tokenize hoặc hiển thị."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_text_for_search(text: str) -> str:
    """Bỏ dấu, lowercase và loại ký tự đặc biệt để so khớp truy vấn nhất quán hơn."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text)


def tokenize(text: str) -> list[str]:
    """Tách text đã normalize thành token đơn giản cho BM25 và matching metadata."""
    normalized = normalize_text_for_search(text)
    if not normalized:
        return []
    return normalized.split(" ")


def truncate_text(text: str, max_chars: int = 400) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def sanitize_chat_history(
    chat_history: Sequence[dict[str, Any]] | None,
    max_messages: int = 10,
    max_chars_per_message: int = 400,
) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    if not chat_history:
        return sanitized

    recent_messages = list(chat_history)[-max_messages:]
    for item in recent_messages:
        if not isinstance(item, dict):
            continue
        role = normalize_whitespace(str(item.get("role") or "")).lower()
        content = truncate_text(str(item.get("content") or ""), max_chars=max_chars_per_message)
        if role not in {"user", "assistant"} or not content:
            continue
        sanitized.append({"role": role, "content": content})
    return sanitized


def format_chat_history(chat_history: Sequence[dict[str, Any]] | None) -> str:
    sanitized = sanitize_chat_history(chat_history)
    if not sanitized:
        return "Khong co lich su hoi thoai truoc do."

    lines: list[str] = []
    for item in sanitized:
        speaker = "Nguoi dung" if item["role"] == "user" else "Tro ly"
        lines.append(f"{speaker}: {item['content']}")
    return "\n".join(lines)


def get_last_chat_message(chat_history: Sequence[dict[str, Any]] | None, role: str | None = None) -> str:
    sanitized = sanitize_chat_history(chat_history)
    for item in reversed(sanitized):
        if role is None or item["role"] == role:
            return item["content"]
    return ""


def match_catalog_values_in_query(query_normalized: str, values: Sequence[str]) -> list[str]:
    """Chỉ match catalog value khi value normalize ra chuỗi hữu ích; tránh empty-string match toàn bộ query."""
    matches: list[str] = []
    padded_query = f" {query_normalized} "
    for value in values:
        normalized_value = normalize_text_for_search(value)
        if not normalized_value or len(normalized_value) < 2:
            continue
        if f" {normalized_value} " in padded_query:
            matches.append(value)
    return matches


def get_movie_title_catalog(index_payload: dict[str, Any]) -> list[str]:
    catalog = index_payload.get("catalog", {})
    explicit_titles = catalog.get("titles") or catalog.get("movie_titles") or []
    if explicit_titles:
        return list(explicit_titles)

    seen: set[str] = set()
    titles: list[str] = []
    for movie in index_payload.get("movies", []):
        for field_name in ("title", "origin_name", "slug"):
            value = normalize_whitespace(str(movie.get(field_name) or ""))
            normalized_value = normalize_text_for_search(value)
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            titles.append(value)
    return sorted(titles, key=normalize_text_for_search)


def first_sentences(text: str, max_sentences: int = 2, max_chars: int = 260) -> str:
    """Lấy phần mở đầu ngắn của nội dung để làm snippet trong câu trả lời hoặc recommendation."""
    text = normalize_whitespace(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    chosen = " ".join(parts[:max_sentences]).strip()
    if len(chosen) <= max_chars:
        return chosen
    return chosen[: max_chars - 1].rstrip() + "…"


def minmax_scale(values: Sequence[float]) -> list[float]:
    """Đưa điểm số về cùng thang 0..1 trước khi trộn sparse score và dense score."""
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.0 if math.isclose(low, 0.0) else 1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _read_utf8_text(path: Path) -> str:
    """Đọc file text UTF-8; fail sớm nếu data nguồn dùng encoding không hỗ trợ."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RAGError(f"Unsupported non-UTF-8 text file in DATA_DIR: {path}") from exc


def _read_pdf_text(path: Path) -> str:
    """Trích text thô từ PDF để đưa vào pipeline chunking giống tài liệu thường."""
    if PdfReader is None:
        raise RAGError("PDF support requires the 'pypdf' package.")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - depends on external files
        raise RAGError(f"Could not open PDF file: {path}") from exc

    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n\n".join(page for page in pages if page).strip()


def _read_docx_text(path: Path) -> str:
    """Đọc nội dung DOCX bằng cách parse XML trong file nén OOXML."""
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("word/document.xml") as document_xml:
                root = ElementTree.parse(document_xml).getroot()
    except KeyError as exc:
        raise RAGError(f"Invalid DOCX file structure: {path}") from exc
    except zipfile.BadZipFile as exc:
        raise RAGError(f"Could not open DOCX file: {path}") from exc

    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", DOCX_NAMESPACE):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{DOCX_NAMESPACE['w']}}}t":
                parts.append(node.text or "")
            elif node.tag == f"{{{DOCX_NAMESPACE['w']}}}tab":
                parts.append("\t")
            elif node.tag == f"{{{DOCX_NAMESPACE['w']}}}br":
                parts.append("\n")
        paragraph_text = "".join(parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return "\n\n".join(paragraphs).strip()


def load_document_text(path: str | Path) -> str:
    """Chọn đúng loader theo extension của tài liệu policy/support."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_utf8_text(path)
    if suffix == ".pdf":
        return _read_pdf_text(path)
    if suffix == ".docx":
        return _read_docx_text(path)
    raise RAGError(f"Unsupported file type: {path.suffix or '<no extension>'}")


def discover_policy_files(data_dir: Path) -> list[Path]:
    """Quét toàn bộ tài liệu hợp lệ trong DATA_DIR để chuẩn bị build corpus policy."""
    if not data_dir.exists():
        raise RAGError(f"DATA_DIR does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise RAGError(f"DATA_DIR is not a directory: {data_dir}")

    return sorted(
        (
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
        ),
        key=lambda path: path.relative_to(data_dir).as_posix().lower(),
    )


def collect_policy_documents(data_dir: str | Path) -> list[SourceDocument]:
    """Nạp tài liệu policy từ đĩa thành cấu trúc nguồn có hash và metadata cơ bản."""
    data_dir = Path(data_dir)
    files = discover_policy_files(data_dir)
    documents: list[SourceDocument] = []
    for path in files:
        relative_path = path.relative_to(data_dir).as_posix()
        text = load_document_text(path).strip()
        if not text:
            continue
        documents.append(
            SourceDocument(
                path=relative_path,
                file_type=path.suffix.lower(),
                text=text,
                hash=sha256_text(text),
                size_bytes=path.stat().st_size,
                characters=len(text),
            )
        )
    return documents


def _clean_lines(text: str) -> list[str]:
    """Giữ nguyên nội dung nhưng bỏ khoảng trắng cuối dòng để tách section ổn định hơn."""
    return [line.rstrip() for line in text.splitlines()]


def split_markdown_sections(text: str, source_name: str) -> list[dict[str, Any]]:
    """Tách markdown theo heading để preserve ngữ cảnh cấu trúc khi chunking."""
    lines = _clean_lines(text)
    sections: list[dict[str, Any]] = []
    h1 = Path(source_name).name
    h2 = ""
    h3 = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        heading_path = [item for item in [h1, h2, h3] if item]
        section_title = " > ".join(heading_path)
        sections.append(
            {
                "title": section_title,
                "section": h3 or h2 or h1,
                "heading_path": heading_path,
                "content": body,
            }
        )

    for line in lines:
        if line.startswith("# "):
            flush()
            buffer = []
            h1 = line[2:].strip() or h1
            h2 = ""
            h3 = ""
            continue
        if line.startswith("## "):
            flush()
            buffer = []
            h2 = line[3:].strip()
            h3 = ""
            continue
        if line.startswith("### "):
            flush()
            buffer = []
            h3 = line[4:].strip()
            continue
        buffer.append(line)

    flush()
    return sections


def split_plain_text_sections(text: str, source_name: str) -> list[dict[str, Any]]:
    """Bao text thường vào một section duy nhất khi file không có heading."""
    body = text.strip()
    if not body:
        return []
    title = Path(source_name).name
    return [
        {
            "title": title,
            "section": title,
            "heading_path": [title],
            "content": body,
        }
    ]


def split_document_sections(text: str, source_name: str, file_type: str) -> list[dict[str, Any]]:
    """Định tuyến sang chiến lược tách section phù hợp với loại file nguồn."""
    if file_type == ".md":
        return split_markdown_sections(text, source_name)
    return split_plain_text_sections(text, source_name)


def chunk_section_text(text: str, max_chars: int = 1100, overlap_chars: int = 150) -> list[str]:
    """Chia section dài thành các chunk có overlap để retrieval không mất ngữ cảnh."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(paragraph):
                break
            start = max(0, end - overlap_chars)
        current = ""

    if current:
        chunks.append(current)
    return chunks


def guess_policy_intent(section_name: str, content: str) -> str:
    """Gán intent gần đúng cho chunk policy để boost retrieval theo loại câu hỏi."""
    text = normalize_text_for_search(f"{section_name} {content}")
    if any(word in text for word in ["hoan tien", "thanh toan", "gia han", "goi"]):
        return "payment"
    if any(word in text for word in ["tre em", "pin", "noi dung", "tuoi"]):
        return "kids"
    if any(word in text for word in ["dang nhap", "mat khau", "tai khoan", "otp"]):
        return "account"
    if any(word in text for word in ["thiet bi", "ho so", "ngoai tuyen", "tai xuong"]):
        return "device"
    if any(word in text for word in ["ho tro", "khieu nai", "email"]):
        return "support"
    return "general"


def build_policy_chunks(documents: Sequence[SourceDocument]) -> list[Chunk]:
    """Biến tài liệu policy thành các chunk có metadata và term frequency để search."""
    chunks: list[Chunk] = []
    for doc_index, document in enumerate(documents, start=1):
        sections = split_document_sections(document.text, document.path, document.file_type)
        for section_index, section in enumerate(sections, start=1):
            section_name = str(section["section"])
            title = str(section["title"])
            heading_path = list(section["heading_path"])
            content = str(section["content"])
            section_chunks = chunk_section_text(content)
            for chunk_index, chunk_text in enumerate(section_chunks, start=1):
                source_kind = "synthetic" if document.path.startswith("generated/") else "primary"
                metadata = {
                    "app_name": "Alpha Cinema",
                    "domain": "policy",
                    "section": section_name,
                    "intent": guess_policy_intent(section_name, chunk_text),
                    "audience": "all_users",
                    "source_file": document.path,
                    "source_kind": source_kind,
                }
                search_text = "\n".join(
                    [
                        f"title: {title}",
                        f"section: {section_name}",
                        f"source: {document.path}",
                        chunk_text,
                    ]
                )
                token_counts = Counter(tokenize(search_text))
                chunks.append(
                    Chunk(
                        chunk_id=f"policy-{doc_index:03d}-{section_index:03d}-{chunk_index:02d}",
                        source=document.path,
                        domain="policy",
                        title=title,
                        section=section_name,
                        heading_path=heading_path,
                        content=chunk_text,
                        metadata=metadata,
                        search_text=search_text,
                        term_freq=dict(token_counts),
                        doc_len=sum(token_counts.values()),
                    )
                )
    return chunks


def firestore_value_to_jsonable(value: Any) -> Any:
    """Đổi kiểu dữ liệu đặc thù của Firestore về kiểu JSON có thể serialize được."""
    if isinstance(value, dict):
        return {key: firestore_value_to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [firestore_value_to_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def init_firestore_client(service_account_path: str | Path):
    """Khởi tạo Firebase app riêng cho service account và trả về Firestore client."""
    if firebase_admin is None or credentials is None or firestore is None:
        raise RAGError("firebase-admin is not installed. Run: pip install -r requirements.txt")

    service_account_path = Path(service_account_path)
    if not service_account_path.exists():
        raise RAGError(f"Firebase service account file not found: {service_account_path}")

    app_name = f"alpha-cinema-{service_account_path.resolve()}"
    existing = next((app for app in firebase_admin._apps.values() if app.name == app_name), None)
    if existing is None:
        cred = credentials.Certificate(str(service_account_path))
        firebase_admin.initialize_app(cred, name=app_name)
    return firestore.client(firebase_admin.get_app(app_name))


def fetch_movies_from_firestore(service_account_path: str | Path, collection_name: str = "movies") -> list[dict[str, Any]]:
    """Đọc toàn bộ collection phim từ Firestore để export sang snapshot JSON."""
    db = init_firestore_client(service_account_path)
    items: list[dict[str, Any]] = []
    try:
        docs = db.collection(collection_name).stream()
    except Exception as exc:  # pragma: no cover - external dependency
        raise RAGError(f"Could not stream Firestore collection '{collection_name}': {exc}") from exc

    for doc in docs:
        payload = firestore_value_to_jsonable(doc.to_dict() or {})
        payload["id"] = doc.id
        items.append(payload)
    return items


def load_movies_from_snapshot(snapshot_path: str | Path | None) -> list[dict[str, Any]]:
    """Nạp snapshot phim đã export; hỗ trợ cả object đơn và list movie."""
    if snapshot_path is None:
        return []
    path = Path(snapshot_path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if isinstance(raw.get("movies"), list):
            raw = raw["movies"]
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raise RAGError(f"Movie snapshot must be a JSON list or object: {path}")
    movies: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            movies.append(item)
    return movies


def normalize_movie(raw_movie: dict[str, Any]) -> MovieDocument:
    """Chuẩn hóa record phim thô thành schema nội bộ phục vụ search và recommendation."""
    def ensure_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))]
        return [normalize_whitespace(str(value))]

    movie_id = normalize_whitespace(str(raw_movie.get("id") or raw_movie.get("slug") or sha256_text(json.dumps(raw_movie, ensure_ascii=False))))
    title = normalize_whitespace(str(raw_movie.get("title") or ""))
    origin_name = normalize_whitespace(str(raw_movie.get("originName") or raw_movie.get("origin_name") or ""))
    slug = normalize_whitespace(str(raw_movie.get("slug") or movie_id))
    year_value = raw_movie.get("year")
    year: int | None = None
    try:
        if year_value not in (None, ""):
            year = int(year_value)
    except Exception:
        year = None

    movie = MovieDocument(
        movie_id=movie_id,
        title=title,
        origin_name=origin_name,
        slug=slug,
        year=year,
        movie_type=normalize_whitespace(str(raw_movie.get("type") or "")),
        status=normalize_whitespace(str(raw_movie.get("status") or "")),
        age_rating=normalize_whitespace(str(raw_movie.get("ageRating") or raw_movie.get("age_rating") or "")),
        categories=ensure_list(raw_movie.get("categories")),
        countries=ensure_list(raw_movie.get("countries")),
        actors=ensure_list(raw_movie.get("actors")),
        directors=ensure_list(raw_movie.get("directors")),
        is_kids_friendly=bool(raw_movie.get("isKidsFriendly", False)),
        content=normalize_whitespace(str(raw_movie.get("content") or "")),
        search_keywords=ensure_list(raw_movie.get("searchKeywords") or raw_movie.get("search_keywords")),
        poster_url=normalize_whitespace(str(raw_movie.get("posterUrl") or raw_movie.get("poster_url") or "")),
        thumb_url=normalize_whitespace(str(raw_movie.get("thumbUrl") or raw_movie.get("thumb_url") or "")),
        modified_time=normalize_whitespace(str(raw_movie.get("modifiedTime") or raw_movie.get("modified_time") or "")),
    )

    fields = [
        f"Tiêu đề: {movie.title}" if movie.title else "",
        f"Tên gốc: {movie.origin_name}" if movie.origin_name else "",
        f"Năm: {movie.year}" if movie.year else "",
        f"Loại: {movie.movie_type}" if movie.movie_type else "",
        f"Trạng thái: {movie.status}" if movie.status else "",
        f"Giới hạn tuổi: {movie.age_rating}" if movie.age_rating else "",
        f"Thể loại: {', '.join(movie.categories)}" if movie.categories else "",
        f"Quốc gia: {', '.join(movie.countries)}" if movie.countries else "",
        f"Đạo diễn: {', '.join(movie.directors)}" if movie.directors else "",
        f"Diễn viên: {', '.join(movie.actors)}" if movie.actors else "",
        f"Từ khóa: {', '.join(movie.search_keywords)}" if movie.search_keywords else "",
        f"Nội dung: {movie.content}" if movie.content else "",
    ]
    movie.search_text = "\n".join(part for part in fields if part)
    token_counts = Counter(tokenize(movie.search_text))
    movie.term_freq = dict(token_counts)
    movie.doc_len = sum(token_counts.values())
    return movie


def split_sentences(text: str) -> list[str]:
    """Tách câu đơn giản để chunk synopsis mượt hơn so với cắt theo ký tự thuần túy."""
    text = normalize_whitespace(text)
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def chunk_synopsis(text: str, max_chars: int = 850, overlap_chars: int = 120) -> list[str]:
    """Chia synopsis phim thành các đoạn ngắn hơn để embedding và retrieval chính xác hơn."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            overlap = current[-overlap_chars:] if current else ""
            current = f"{overlap} {sentence}".strip() if overlap else sentence
        else:
            start = 0
            while start < len(sentence):
                end = min(start + max_chars, len(sentence))
                piece = sentence[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(sentence):
                    break
                start = max(0, end - overlap_chars)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def build_movie_chunks(movies: Sequence[MovieDocument]) -> list[Chunk]:
    """Biến metadata phim và synopsis thành chunk movie để QA có thể truy xuất theo nội dung."""
    chunks: list[Chunk] = []
    for movie_index, movie in enumerate(movies, start=1):
        base_metadata = {
            "app_name": "Alpha Cinema",
            "domain": "movie",
            "movie_id": movie.movie_id,
            "title": movie.title,
            "origin_name": movie.origin_name,
            "slug": movie.slug,
            "year": movie.year,
            "type": movie.movie_type,
            "status": movie.status,
            "age_rating": movie.age_rating,
            "categories": movie.categories,
            "countries": movie.countries,
            "actors": movie.actors,
            "directors": movie.directors,
            "is_kids_friendly": movie.is_kids_friendly,
            "poster_url": movie.poster_url,
            "thumb_url": movie.thumb_url,
            "modified_time": movie.modified_time,
            "intent": "movie",
            "audience": "kids" if movie.is_kids_friendly else "general",
        }

        overview = "\n".join(
            part
            for part in [
                f"Tiêu đề: {movie.title}" if movie.title else "",
                f"Tên gốc: {movie.origin_name}" if movie.origin_name else "",
                f"Năm: {movie.year}" if movie.year else "",
                f"Loại: {movie.movie_type}" if movie.movie_type else "",
                f"Trạng thái: {movie.status}" if movie.status else "",
                f"Giới hạn tuổi: {movie.age_rating}" if movie.age_rating else "",
                f"Thể loại: {', '.join(movie.categories)}" if movie.categories else "",
                f"Quốc gia: {', '.join(movie.countries)}" if movie.countries else "",
            ]
            if part
        )
        cast_block = "\n".join(
            part
            for part in [
                f"Đạo diễn: {', '.join(movie.directors)}" if movie.directors else "",
                f"Diễn viên: {', '.join(movie.actors)}" if movie.actors else "",
                f"Từ khóa tìm kiếm: {', '.join(movie.search_keywords)}" if movie.search_keywords else "",
            ]
            if part
        )

        sections = [
            ("overview", overview),
            ("cast", cast_block),
        ]
        synopsis_chunks = chunk_synopsis(movie.content) if movie.content else []
        for index, synopsis in enumerate(synopsis_chunks, start=1):
            sections.append((f"synopsis_{index}", synopsis))

        movie.chunk_ids = []
        for section_index, (section_name, chunk_text) in enumerate(sections, start=1):
            if not chunk_text:
                continue
            search_text = "\n".join(
                [
                    f"section: {section_name}",
                    movie.search_text,
                    chunk_text,
                ]
            )
            token_counts = Counter(tokenize(search_text))
            chunk_id = f"movie-{movie_index:04d}-{section_index:02d}"
            movie.chunk_ids.append(chunk_id)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=f"firestore:{movie.slug or movie.movie_id}",
                    domain="movie",
                    title=movie.title or movie.slug or movie.movie_id,
                    section=section_name,
                    heading_path=[movie.title or movie.slug or movie.movie_id, section_name],
                    content=chunk_text,
                    metadata={**base_metadata, "section": section_name},
                    search_text=search_text,
                    term_freq=dict(token_counts),
                    doc_len=sum(token_counts.values()),
                )
            )
    return chunks


def build_doc_freq(items: Sequence[dict[str, Any] | Chunk | MovieDocument], field_name: str = "term_freq") -> dict[str, int]:
    """Tính document frequency cho BM25 từ tập chunk hoặc movie document."""
    doc_freq: defaultdict[str, int] = defaultdict(int)
    for item in items:
        mapping = getattr(item, field_name) if hasattr(item, field_name) else item.get(field_name, {})
        for term in mapping:
            doc_freq[term] += 1
    return dict(doc_freq)


def is_openai_model(model: str) -> bool:
    """Nhận diện model thuộc họ OpenAI để chọn SDK và env key tương ứng."""
    normalized = (model or "").strip().lower()
    return normalized.startswith(("gpt-", "text-embedding-", "o1", "o3", "o4"))


def is_gemini_model(model: str) -> bool:
    """Nhận diện model Gemini để route sang client của Google."""
    normalized = (model or "").strip().lower()
    return normalized.startswith(("gemini", "models/gemini"))


def resolve_api_key_for_model(model: str, api_key: str | None = None) -> str | None:
    """Chọn API key phù hợp với provider của model đang dùng."""
    if is_openai_model(model):
        return api_key or os.getenv("OPENAI_API_KEY")
    if is_gemini_model(model):
        return api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def get_gemini_client(api_key: str | None = None):
    """Tạo Gemini client và báo lỗi rõ ràng nếu thiếu dependency hoặc API key."""
    if genai is None or types is None:
        raise RAGError("google-genai is not installed. Run: pip install -r requirements.txt")
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RAGError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    return genai.Client(api_key=api_key)


def get_openai_client(api_key: str | None = None):
    """Tạo OpenAI client dùng cho embedding hoặc generation."""
    if OpenAI is None:
        raise RAGError("openai is not installed. Run: pip install -r requirements.txt")
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RAGError("Missing OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def generate_text_with_model(
    prompt: str,
    api_key: str | None,
    model: str,
    temperature: float = 0.2,
) -> str | None:
    resolved_api_key = resolve_api_key_for_model(model, api_key)
    if not resolved_api_key:
        return None

    if is_openai_model(model):
        client = get_openai_client(resolved_api_key)
        provider = "OpenAI"
    else:
        client = get_gemini_client(resolved_api_key)
        provider = "Gemini"

    try:
        try:
            if is_openai_model(model):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature, top_p=0.9),
                )
        except (ClientError, OpenAIAPIError, RAGError) as exc:
            print(f"Warning: {provider} text generation is unavailable. Reason: {exc}")
            return None
    finally:
        try:
            client.close()
        except Exception:
            pass

    if is_openai_model(model):
        choice = ((getattr(response, "choices", None) or [])[:1] or [None])[0]
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "").strip()
    else:
        text = (getattr(response, "text", None) or "").strip()
    cleaned_text = strip_source_annotations(text)
    return cleaned_text or None


def _extract_gemini_embedding_values(response: object) -> list[list[float]]:
    """Chuẩn hóa output embedding của Gemini về list vector đơn giản."""
    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        return [list(item.values) for item in embeddings]

    embedding = getattr(response, "embedding", None)
    if embedding:
        return [list(embedding.values)]

    raise RAGError("Could not read embeddings from Gemini response.")


def _extract_openai_embedding_values(response: object) -> list[list[float]]:
    """Chuẩn hóa output embedding của OpenAI về list vector đơn giản."""
    data = getattr(response, "data", None) or []
    vectors = [list(item.embedding) for item in data if getattr(item, "embedding", None) is not None]
    if vectors:
        return vectors
    raise RAGError("Could not read embeddings from OpenAI response.")


def _is_insufficient_quota_error(exc: Exception) -> bool:
    """Phân biệt lỗi hết quota cứng với lỗi rate limit tạm thời để xử lý retry hợp lý."""
    message = str(exc).lower()
    return "insufficient_quota" in message or "exceeded your current quota" in message


def _is_retryable_embedding_error(exc: Exception) -> bool:
    """Đánh dấu các lỗi embedding có thể retry sau một khoảng delay."""
    if _is_insufficient_quota_error(exc):
        return False
    message = str(exc).upper()
    return isinstance(exc, OpenAIRateLimitError) or "429" in message or "RESOURCE_EXHAUSTED" in message or "RATE LIMIT" in message


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    """Đọc retry delay nếu provider trả về gợi ý thời gian chờ trong message lỗi."""
    message = str(exc)
    patterns = (
        r"Please retry in\s+(\d+(?:\.\d+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def embed_texts(
    texts: list[str],
    api_key: str | None = None,
    model: str = "text-embedding-3-small",
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = 768,
    batch_size: int = 64,
    requests_per_minute_limit: int = 90,
    max_retries: int = 6,
) -> list[list[float]]:
    """Sinh dense embeddings theo batch, có throttle và retry để tránh lỗi quota tạm thời."""
    if not texts:
        return []
    if batch_size <= 0:
        raise RAGError("Embedding batch size must be greater than 0.")
    if max_retries < 0:
        raise RAGError("Embedding max retries must be greater than or equal to 0.")

    resolved_api_key = resolve_api_key_for_model(model, api_key)
    if is_openai_model(model):
        client = get_openai_client(resolved_api_key)
        provider = "OpenAI"
    else:
        client = get_gemini_client(resolved_api_key)
        provider = "Gemini"
    vectors: list[list[float]] = []
    min_interval_seconds = 60.0 / requests_per_minute_limit if requests_per_minute_limit > 0 else 0.0
    last_request_at: float | None = None

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            attempt = 0
            while True:
                if min_interval_seconds > 0 and last_request_at is not None:
                    elapsed = time.monotonic() - last_request_at
                    if elapsed < min_interval_seconds:
                        time.sleep(min_interval_seconds - elapsed)
                last_request_at = time.monotonic()
                try:
                    if is_openai_model(model):
                        request_kwargs: dict[str, Any] = {
                            "model": model,
                            "input": batch,
                        }
                        if model.startswith("text-embedding-3") and output_dimensionality > 0:
                            request_kwargs["dimensions"] = output_dimensionality
                        response = client.embeddings.create(**request_kwargs)
                    else:
                        response = client.models.embed_content(
                            model=model,
                            contents=batch,
                            config=types.EmbedContentConfig(
                                task_type=task_type,
                                output_dimensionality=output_dimensionality,
                            ),
                        )
                    break
                except (ClientError, OpenAIAPIError) as exc:
                    if _is_retryable_embedding_error(exc) and attempt < max_retries:
                        retry_delay = _extract_retry_delay_seconds(exc) or min_interval_seconds or 5.0
                        retry_delay = max(retry_delay, 1.0)
                        batch_index = (start // batch_size) + 1
                        total_batches = math.ceil(len(texts) / batch_size)
                        print(
                            f"{provider} embeddings hit a temporary rate limit. "
                            f"Retrying batch {batch_index}/{total_batches} in {retry_delay:.1f}s..."
                        )
                        time.sleep(retry_delay)
                        attempt += 1
                        continue
                    if _is_retryable_embedding_error(exc):
                        raise RAGError(
                            f"{provider} API rate limit was still exhausted after retries. "
                            "Try again later or lower EMBEDDING_REQUESTS_PER_MINUTE / "
                            "increase EMBEDDING_BATCH_SIZE. "
                            f"Last error: {exc}"
                        ) from exc
                    raise RAGError(f"{provider} API error while creating embeddings: {exc}") from exc
            if is_openai_model(model):
                vectors.extend(_extract_openai_embedding_values(response))
            else:
                vectors.extend(_extract_gemini_embedding_values(response))
    finally:
        try:
            client.close()
        except Exception:
            pass

    return vectors


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector về độ dài 1 để tính cosine similarity bằng dot product."""
    norm = float(np.linalg.norm(vector))
    if math.isclose(norm, 0.0):
        return vector
    return vector / norm


def maybe_build_dense_vectors(
    chunk_texts: list[str],
    movie_texts: list[str],
    api_key: str | None,
    embedding_model: str,
    output_dimensionality: int,
    embedding_batch_size: int,
    embedding_requests_per_minute: int,
    embedding_max_retries: int,
) -> tuple[list[list[float]] | None, list[list[float]] | None]:
    """Sinh vector cho chunk và movie nếu có API key; nếu không thì cho phép chạy sparse-only."""
    resolved_api_key = resolve_api_key_for_model(embedding_model, api_key)
    if not resolved_api_key:
        return None, None
    chunk_vectors = embed_texts(
        chunk_texts,
        api_key=resolved_api_key,
        model=embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=output_dimensionality,
        batch_size=embedding_batch_size,
        requests_per_minute_limit=embedding_requests_per_minute,
        max_retries=embedding_max_retries,
    )
    movie_vectors = embed_texts(
        movie_texts,
        api_key=resolved_api_key,
        model=embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=output_dimensionality,
        batch_size=embedding_batch_size,
        requests_per_minute_limit=embedding_requests_per_minute,
        max_retries=embedding_max_retries,
    )
    return chunk_vectors, movie_vectors


def build_source_manifest(data_dir: str | Path, documents: Sequence[SourceDocument], movies_count: int) -> dict[str, Any]:
    """Tạo metadata nguồn đầu vào để biết index được build từ tập dữ liệu nào."""
    fingerprint_parts = [f"{doc.path}\t{doc.hash}" for doc in documents]
    fingerprint_parts.append(f"movies\t{movies_count}")
    return {
        "data_dir": str(Path(data_dir)),
        "file_count": len(documents),
        "source_files": [
            {
                "path": doc.path,
                "file_type": doc.file_type,
                "hash": doc.hash,
                "size_bytes": doc.size_bytes,
                "characters": doc.characters,
            }
            for doc in documents
        ],
        "source_hash": sha256_text("\n".join(fingerprint_parts)),
        "total_characters": sum(doc.characters for doc in documents),
        "movie_count": movies_count,
    }


def build_index_payload(
    data_dir: str | Path,
    movies_snapshot_path: str | Path | None = None,
    api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    output_dimensionality: int = 768,
    embedding_batch_size: int = 64,
    embedding_requests_per_minute: int = 90,
    embedding_max_retries: int = 6,
    allow_sparse_fallback_on_embedding_error: bool = True,
) -> dict[str, Any]:
    """Build index hoàn chỉnh gồm chunks, BM25 stats, catalog và dense vectors nếu có thể."""
    policy_documents = collect_policy_documents(data_dir)
    raw_movies = load_movies_from_snapshot(movies_snapshot_path)
    movies = [normalize_movie(item) for item in raw_movies]

    if not policy_documents and not movies:
        raise RAGError("No policy documents or movie data found to build the index.")

    policy_chunks = build_policy_chunks(policy_documents)
    movie_chunks = build_movie_chunks(movies)
    all_chunks = policy_chunks + movie_chunks
    if not all_chunks:
        raise RAGError("No chunks were created from the provided data.")

    chunk_df = build_doc_freq(all_chunks, field_name="term_freq")
    chunk_avg_len = sum(chunk.doc_len for chunk in all_chunks) / max(len(all_chunks), 1)
    movie_df = build_doc_freq(movies, field_name="term_freq")
    movie_avg_len = sum(movie.doc_len for movie in movies) / max(len(movies), 1) if movies else 0.0

    chunk_texts = [chunk.search_text for chunk in all_chunks]
    movie_texts = [movie.search_text for movie in movies]
    dense_vectors_error: str | None = None
    try:
        chunk_vectors, movie_vectors = maybe_build_dense_vectors(
            chunk_texts=chunk_texts,
            movie_texts=movie_texts,
            api_key=api_key,
            embedding_model=embedding_model,
            output_dimensionality=output_dimensionality,
            embedding_batch_size=embedding_batch_size,
            embedding_requests_per_minute=embedding_requests_per_minute,
            embedding_max_retries=embedding_max_retries,
        )
    except RAGError as exc:
        if not allow_sparse_fallback_on_embedding_error:
            raise
        dense_vectors_error = str(exc)
        chunk_vectors, movie_vectors = None, None
        print(
            "Warning: dense embeddings are unavailable, continuing with sparse-only index build. "
            f"Reason: {exc}"
        )

    manifest = build_source_manifest(data_dir, policy_documents, len(movies))

    catalog = {
        "categories": sorted({category for movie in movies for category in movie.categories}),
        "countries": sorted({country for movie in movies for country in movie.countries}),
        "actors": sorted({actor for movie in movies for actor in movie.actors}),
        "directors": sorted({director for movie in movies for director in movie.directors}),
        "age_ratings": sorted({rating for movie in movies for rating in [movie.age_rating] if rating}),
        "titles": sorted(
            {
                value
                for movie in movies
                for value in [movie.title, movie.origin_name, movie.slug]
                if normalize_whitespace(value)
            },
            key=normalize_text_for_search,
        ),
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "app_name": "Alpha Cinema",
        "data_dir": manifest["data_dir"],
        "file_count": manifest["file_count"],
        "movie_count": manifest["movie_count"],
        "source_files": manifest["source_files"],
        "source_hash": manifest["source_hash"],
        "total_characters": manifest["total_characters"],
        "embedding_model": embedding_model if chunk_vectors is not None else None,
        "output_dimensionality": output_dimensionality if chunk_vectors is not None else None,
        "has_dense_vectors": chunk_vectors is not None,
        "dense_vectors_error": dense_vectors_error,
        "chunk_count": len(all_chunks),
        "chunks": [asdict(chunk) for chunk in all_chunks],
        "chunk_vectors": chunk_vectors,
        "chunk_bm25": {
            "doc_count": len(all_chunks),
            "avg_doc_len": chunk_avg_len,
            "doc_freq": chunk_df,
        },
        "movies": [asdict(movie) for movie in movies],
        "movie_vectors": movie_vectors,
        "movie_bm25": {
            "doc_count": len(movies),
            "avg_doc_len": movie_avg_len,
            "doc_freq": movie_df,
        },
        "catalog": catalog,
    }


def save_index(index_payload: dict[str, Any], output_path: str | Path) -> Path:
    """Lưu index đã build ra JSON để app.py có thể nạp lại khi chạy API."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dumps(index_payload), encoding="utf-8")
    return output_path


def load_index(index_path: str | Path) -> dict[str, Any]:
    """Đọc index JSON từ đĩa vào memory."""
    return json.loads(Path(index_path).read_text(encoding="utf-8"))


def bm25_score(
    query_tokens: Sequence[str],
    term_freq: dict[str, int],
    doc_len: int,
    doc_freq: dict[str, int],
    doc_count: int,
    avg_doc_len: float,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """Tính sparse relevance score theo BM25 cho một document/chunk."""
    if not query_tokens or not term_freq or doc_count <= 0:
        return 0.0
    if avg_doc_len <= 0:
        avg_doc_len = 1.0

    score = 0.0
    for term in query_tokens:
        tf = term_freq.get(term, 0)
        if tf <= 0:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * numerator / max(denominator, 1e-9)
    return score


def embed_query(query: str, api_key: str | None, model: str, output_dimensionality: int) -> np.ndarray:
    """Embedding riêng cho câu hỏi của người dùng để so sánh với vector đã index."""
    vectors = embed_texts(
        texts=[query],
        api_key=api_key,
        model=model,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=output_dimensionality,
        batch_size=1,
    )
    return normalize_vector(np.array(vectors[0], dtype=np.float32))


def detect_intent(query: str) -> str:
    """Ước lượng truy vấn thuộc policy, movie, recommendation hay mixed."""
    normalized = normalize_text_for_search(query)
    recommendation_keywords = [
        "goi y",
        "de xuat",
        "nen xem",
        "recommend",
        "tuong tu",
        "hay cho toi phim",
        "liet ke",
        "danh sach",
        "top phim",
    ]
    policy_keywords = [
        "goi",
        "thanh toan",
        "hoan tien",
        "gia han",
        "huy",
        "tai khoan",
        "mat khau",
        "dang nhap",
        "thiet bi",
        "tre em",
        "pin",
        "ho tro",
        "hoa don",
        "giao dich",
    ]
    movie_keywords = [
        "phim",
        "dien vien",
        "dao dien",
        "the loai",
        "noi dung",
        "tap",
        "series",
        "movie",
        "nam",
        "trung quoc",
        "han quoc",
        "viet nam",
    ]

    if any(keyword in normalized for keyword in recommendation_keywords):
        return "recommendation"
    if "phim" in normalized and any(keyword in normalized for keyword in ["dang hot", "hot hien nay", "noi bat", "trending"]):
        return "recommendation"

    policy_score = sum(1 for keyword in policy_keywords if keyword in normalized)
    movie_score = sum(1 for keyword in movie_keywords if keyword in normalized)

    if policy_score and movie_score:
        return "mixed"
    if policy_score:
        return "policy"
    if movie_score:
        return "movie"
    return "mixed"


def collect_query_preferences(query: str, index_payload: dict[str, Any]) -> dict[str, Any]:
    """Trích metadata người dùng đang nhắc tới để boost recommendation và retrieval."""
    normalized = normalize_text_for_search(query)
    years = [int(match) for match in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
    catalog = index_payload.get("catalog", {})

    matched_categories = match_catalog_values_in_query(normalized, catalog.get("categories", []))
    matched_countries = match_catalog_values_in_query(normalized, catalog.get("countries", []))
    matched_actors = match_catalog_values_in_query(normalized, catalog.get("actors", []))
    matched_directors = match_catalog_values_in_query(normalized, catalog.get("directors", []))
    matched_age_ratings = match_catalog_values_in_query(normalized, catalog.get("age_ratings", []))
    matched_titles = match_catalog_values_in_query(normalized, get_movie_title_catalog(index_payload))

    wants_kids = any(term in normalized for term in ["tre em", "thieu nhi", "kid", "kids", "be", "gia dinh"])
    wants_series = any(term in normalized for term in ["series", "phim bo", "tv series"])
    wants_movie = any(term in normalized for term in ["phim le", "movie", "dien anh"])
    wants_completed = any(term in normalized for term in ["hoan thanh", "completed", "full"])
    wants_hot = any(term in normalized for term in ["hot", "noi bat", "xu huong", "trending", "dang hot"])

    return {
        "normalized_query": normalized,
        "years": years,
        "categories": matched_categories,
        "countries": matched_countries,
        "actors": matched_actors,
        "directors": matched_directors,
        "age_ratings": matched_age_ratings,
        "titles": matched_titles,
        "wants_kids": wants_kids,
        "wants_series": wants_series,
        "wants_movie": wants_movie,
        "wants_completed": wants_completed,
        "wants_hot": wants_hot,
    }


def is_generic_hot_query(preferences: dict[str, Any]) -> bool:
    """Truy vấn 'phim hot' chung, không kèm filter cụ thể, nên ưu tiên ranking theo độ mới/trạng thái."""
    return bool(
        preferences.get("wants_hot")
        and not preferences.get("years")
        and not preferences.get("categories")
        and not preferences.get("countries")
        and not preferences.get("actors")
        and not preferences.get("directors")
        and not preferences.get("age_ratings")
        and not preferences.get("wants_kids")
        and not preferences.get("wants_movie")
        and not preferences.get("wants_completed")
    )


def is_followup_question(question: str, chat_history: Sequence[dict[str, Any]] | None = None) -> bool:
    normalized = normalize_text_for_search(question)
    if not normalized or not sanitize_chat_history(chat_history):
        return False

    padded = f" {normalized} "
    followup_phrases = [
        " phim do ",
        " bo do ",
        " goi do ",
        " cai do ",
        " cai nay ",
        " cai kia ",
        " neu vay ",
        " the con ",
        " con phim ",
        " con goi ",
        " con tai khoan ",
    ]
    if any(phrase in padded for phrase in followup_phrases):
        return True
    if normalized.startswith(("con ", "the con", "neu vay", "vay ", "the ", "roi ")):
        return True
    return len(tokenize(question)) <= 4 and detect_intent(question) == "mixed"


def merge_query_preferences(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalized_query": current.get("normalized_query") or previous.get("normalized_query") or "",
        "years": current.get("years") or previous.get("years") or [],
        "categories": current.get("categories") or previous.get("categories") or [],
        "countries": current.get("countries") or previous.get("countries") or [],
        "actors": current.get("actors") or previous.get("actors") or [],
        "directors": current.get("directors") or previous.get("directors") or [],
        "age_ratings": current.get("age_ratings") or previous.get("age_ratings") or [],
        "titles": current.get("titles") or previous.get("titles") or [],
        "wants_kids": bool(current.get("wants_kids") or previous.get("wants_kids")),
        "wants_series": bool(current.get("wants_series") or previous.get("wants_series")),
        "wants_movie": bool(current.get("wants_movie") or previous.get("wants_movie")),
        "wants_completed": bool(current.get("wants_completed") or previous.get("wants_completed")),
        "wants_hot": bool(current.get("wants_hot") or previous.get("wants_hot")),
    }


def build_structured_movie_query(intent: str, preferences: dict[str, Any]) -> str:
    if preferences.get("titles"):
        title = preferences["titles"][0]
        if intent == "recommendation":
            return f"Goi y phim tuong tu voi {title}"
        return f"Thong tin ve phim {title}"

    verb = "Goi y phim" if intent == "recommendation" else "Tim phim"
    parts: list[str] = []
    if preferences.get("categories"):
        parts.append(f"the loai {', '.join(preferences['categories'][:2])}")
    if preferences.get("countries"):
        parts.append(f"quoc gia {', '.join(preferences['countries'][:2])}")
    if preferences.get("actors"):
        parts.append(f"co dien vien {', '.join(preferences['actors'][:2])}")
    if preferences.get("directors"):
        parts.append(f"do {', '.join(preferences['directors'][:2])} dao dien")
    if preferences.get("years"):
        parts.append(f"nam {', '.join(str(year) for year in preferences['years'][:2])}")
    if preferences.get("age_ratings"):
        parts.append(f"gioi han tuoi {', '.join(preferences['age_ratings'][:2])}")
    if preferences.get("wants_kids"):
        parts.append("phu hop tre em")
    if preferences.get("wants_series"):
        parts.append("dang phim bo")
    if preferences.get("wants_movie"):
        parts.append("dang phim le")
    if preferences.get("wants_completed"):
        parts.append("da hoan thanh")
    if preferences.get("wants_hot"):
        parts.append("dang hot")
    if not parts:
        return ""
    return f"{verb} " + ", ".join(parts)


def heuristic_contextualize_question(
    question: str,
    chat_history: Sequence[dict[str, Any]] | None,
    index_payload: dict[str, Any],
) -> str:
    question = normalize_whitespace(question)
    sanitized_history = sanitize_chat_history(chat_history)
    if not sanitized_history or not is_followup_question(question, sanitized_history):
        return question

    previous_user = get_last_chat_message(sanitized_history, role="user")
    previous_assistant = get_last_chat_message(sanitized_history, role="assistant")
    previous_context = previous_user or previous_assistant
    if not previous_context:
        return question

    current_preferences = collect_query_preferences(question, index_payload)
    previous_preferences = collect_query_preferences(previous_context, index_payload)
    if not previous_preferences.get("titles") and previous_assistant:
        assistant_preferences = collect_query_preferences(previous_assistant, index_payload)
        if len(assistant_preferences.get("titles", [])) == 1:
            previous_preferences["titles"] = assistant_preferences["titles"]
    merged_preferences = merge_query_preferences(previous_preferences, current_preferences)

    previous_intent = detect_intent(previous_context)
    current_intent = detect_intent(question)
    effective_intent = current_intent
    if current_intent == "mixed" and previous_intent != "mixed":
        effective_intent = previous_intent

    if effective_intent in {"movie", "recommendation"}:
        structured_query = build_structured_movie_query(effective_intent, merged_preferences)
        if structured_query:
            return structured_query

    return normalize_whitespace(f"{previous_context}. {question}")


def extract_requested_result_count(query: str, default_count: int, max_count: int) -> int:
    """Đọc số lượng user muốn liệt kê từ query, ví dụ 'kể 2 phim' hoặc 'top 3'."""
    match = re.search(r"\b([1-9]|10)\b", query)
    if match:
        return max(1, min(int(match.group(1)), max_count))

    normalized = normalize_text_for_search(query)
    word_to_number = {
        "mot": 1,
        "hai": 2,
        "ba": 3,
        "bon": 4,
        "tu": 4,
        "nam": 5,
        "sau": 6,
        "bay": 7,
        "tam": 8,
        "chin": 9,
        "muoi": 10,
    }
    for word, number in word_to_number.items():
        if re.search(rf"\b{word}\b", normalized):
            return max(1, min(number, max_count))
    return max(1, min(default_count, max_count))


def movie_metadata_boost(movie: dict[str, Any], preferences: dict[str, Any]) -> tuple[float, list[str]]:
    """Tính điểm cộng dựa trên metadata phim khớp với preference trong query."""
    boost = 0.0
    reasons: list[str] = []

    movie_categories = {normalize_text_for_search(item) for item in movie.get("categories", [])}
    movie_countries = {normalize_text_for_search(item) for item in movie.get("countries", [])}
    movie_actors = {normalize_text_for_search(item) for item in movie.get("actors", [])}
    movie_directors = {normalize_text_for_search(item) for item in movie.get("directors", [])}
    movie_titles = {
        normalize_text_for_search(str(value))
        for value in [movie.get("title"), movie.get("origin_name"), movie.get("slug")]
        if normalize_text_for_search(str(value))
    }
    movie_age = normalize_text_for_search(movie.get("age_rating") or movie.get("ageRating") or "")
    movie_type = normalize_text_for_search(movie.get("movie_type") or movie.get("type") or "")
    movie_status = normalize_text_for_search(movie.get("status") or "")
    year = movie.get("year")
    kids_flag = bool(movie.get("is_kids_friendly") or movie.get("isKidsFriendly"))
    modified_time_raw = str(movie.get("modified_time") or "")
    modified_at: datetime | None = None
    if modified_time_raw:
        try:
            modified_at = datetime.fromisoformat(modified_time_raw.replace("Z", "+00:00"))
        except ValueError:
            modified_at = None

    for category in preferences["categories"]:
        if normalize_text_for_search(category) in movie_categories:
            boost += 1.8
            reasons.append(f"khớp thể loại {category}")
    for country in preferences["countries"]:
        if normalize_text_for_search(country) in movie_countries:
            boost += 1.5
            reasons.append(f"khớp quốc gia {country}")
    for actor in preferences["actors"]:
        if normalize_text_for_search(actor) in movie_actors:
            boost += 2.8
            reasons.append(f"có diễn viên {actor}")
    for director in preferences["directors"]:
        if normalize_text_for_search(director) in movie_directors:
            boost += 2.3
            reasons.append(f"do {director} đạo diễn")
    for title in preferences.get("titles", []):
        if normalize_text_for_search(title) in movie_titles:
            boost += 4.0
            reasons.append(f"dung phim {title}")
    for rating in preferences["age_ratings"]:
        if normalize_text_for_search(rating) == movie_age:
            boost += 1.0
            reasons.append(f"khớp giới hạn tuổi {rating}")
    if preferences["years"]:
        if year in preferences["years"]:
            boost += 1.6
            reasons.append(f"đúng năm {year}")
        else:
            boost -= 0.6
    if preferences["wants_kids"] and kids_flag:
        boost += 2.0
        reasons.append("phù hợp trẻ em")
    if preferences["wants_series"] and movie_type == "series":
        boost += 1.2
        reasons.append("là phim bộ/series")
    if preferences["wants_movie"] and movie_type in {"movie", "single"}:
        boost += 1.2
        reasons.append("là phim lẻ")
    if preferences["wants_completed"] and movie_status == "completed":
        boost += 1.0
        reasons.append("đã hoàn thành")
    if preferences.get("wants_hot"):
        current_year = datetime.now(timezone.utc).year
        if isinstance(year, int):
            if year >= current_year:
                boost += 1.2
                reasons.append(f"phim mới ({year})")
            elif year >= current_year - 1:
                boost += 0.7
                reasons.append(f"phát hành gần đây ({year})")
        if movie_status == "ongoing":
            boost += 0.9
            reasons.append("đang cập nhật")
        if modified_at is not None:
            age_days = (datetime.now(timezone.utc) - modified_at).days
            if age_days <= 14:
                boost += 0.8
                reasons.append("cập nhật gần đây")
            elif age_days <= 45:
                boost += 0.4
                reasons.append("mới cập nhật")

    return boost, reasons


def chunk_metadata_boost(chunk: dict[str, Any], preferences: dict[str, Any]) -> float:
    """Áp dụng boost metadata cho chunk, đặc biệt hữu ích với policy intent heuristics."""
    metadata = chunk.get("metadata", {})
    boost, _ = movie_metadata_boost(metadata, preferences)
    if chunk.get("domain") == "policy":
        query = preferences["normalized_query"]
        section = normalize_text_for_search(str(metadata.get("section") or chunk.get("section") or ""))
        if metadata.get("source_kind") == "synthetic":
            boost -= 0.35
        if "hoan tien" in query and "payment" == metadata.get("intent"):
            boost += 0.8
        if "thiet bi" in query and metadata.get("intent") == "device":
            boost += 0.8
        if ("tre em" in query or "pin" in query) and metadata.get("intent") == "kids":
            boost += 0.8
        if ("mat khau" in query or "dang nhap" in query) and metadata.get("intent") == "account":
            boost += 0.8
        if section and section in query:
            boost += 0.4
    return boost


def filter_candidate_indices(chunks: list[dict[str, Any]], intent: str) -> list[int]:
    """Giảm không gian tìm kiếm bằng cách lọc chunk theo domain phù hợp với intent."""
    if intent == "policy":
        return [idx for idx, chunk in enumerate(chunks) if chunk.get("domain") == "policy"]
    if intent in {"movie", "recommendation"}:
        return [idx for idx, chunk in enumerate(chunks) if chunk.get("domain") == "movie"]
    return list(range(len(chunks)))


def retrieve(query: str, index_payload: dict[str, Any], api_key: str | None = None, top_k: int = 6) -> list[dict[str, Any]]:
    """Hybrid retrieval: lọc theo intent, tính BM25, cộng metadata boost và thêm dense score nếu có."""
    query = query.strip()
    if not query:
        return []

    intent = detect_intent(query)
    preferences = collect_query_preferences(query, index_payload)
    query_tokens = tokenize(query)
    chunks = index_payload.get("chunks", [])
    if not chunks:
        return []

    candidate_indices = filter_candidate_indices(chunks, intent)
    if not candidate_indices:
        candidate_indices = list(range(len(chunks)))

    chunk_bm25 = index_payload.get("chunk_bm25", {})
    doc_freq = chunk_bm25.get("doc_freq", {})
    doc_count = int(chunk_bm25.get("doc_count", len(chunks)))
    avg_doc_len = float(chunk_bm25.get("avg_doc_len", 1.0))

    sparse_scores: list[float] = []
    dense_scores: list[float] = [0.0 for _ in candidate_indices]
    metadata_boosts: list[float] = []

    for idx in candidate_indices:
        chunk = chunks[idx]
        sparse_scores.append(
            bm25_score(
                query_tokens,
                chunk.get("term_freq", {}),
                int(chunk.get("doc_len", 0)),
                doc_freq,
                doc_count,
                avg_doc_len,
            )
        )
        metadata_boosts.append(chunk_metadata_boost(chunk, preferences))

    embedding_model = index_payload.get("embedding_model") or "text-embedding-3-small"
    resolved_api_key = resolve_api_key_for_model(embedding_model, api_key)
    has_dense = bool(index_payload.get("has_dense_vectors")) and resolved_api_key and index_payload.get("chunk_vectors")
    if has_dense:
        try:
            query_vector = embed_query(
                query,
                api_key=resolved_api_key,
                model=embedding_model,
                output_dimensionality=int(index_payload.get("output_dimensionality") or 768),
            )
            matrix = np.array([index_payload["chunk_vectors"][idx] for idx in candidate_indices], dtype=np.float32)
            matrix = np.vstack([normalize_vector(row) for row in matrix])
            dense_scores = list((matrix @ query_vector).astype(float))
        except Exception:
            dense_scores = [0.0 for _ in candidate_indices]

    sparse_norm = minmax_scale(sparse_scores)
    dense_norm = minmax_scale(dense_scores)

    results: list[RetrievalResult] = []
    for local_rank, idx in enumerate(candidate_indices):
        chunk = chunks[idx]
        source_kind = chunk.get("metadata", {}).get("source_kind")
        source_weight = 0.82 if source_kind == "synthetic" else 1.0
        final_score = (0.65 * sparse_norm[local_rank] + 0.35 * dense_norm[local_rank]) * source_weight + metadata_boosts[local_rank]
        results.append(
            RetrievalResult(
                rank=0,
                score=float(final_score),
                sparse_score=float(sparse_scores[local_rank]),
                dense_score=float(dense_scores[local_rank]),
                metadata_boost=float(metadata_boosts[local_rank]),
                chunk_id=str(chunk.get("chunk_id")),
                source=str(chunk.get("source")),
                domain=str(chunk.get("domain")),
                title=str(chunk.get("title")),
                section=str(chunk.get("section")),
                heading_path=list(chunk.get("heading_path", [])),
                content=str(chunk.get("content")),
                metadata=dict(chunk.get("metadata", {})),
            )
        )

    ranked = sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        record = asdict(item)
        record["rank"] = rank
        output.append(record)
    return output


def recommend_movies(query: str, index_payload: dict[str, Any], api_key: str | None = None, top_n: int = 5) -> list[dict[str, Any]]:
    """Xếp hạng phim theo query bằng sparse score, dense score và metadata matching."""
    movies = index_payload.get("movies", [])
    if not movies:
        return []

    preferences = collect_query_preferences(query, index_payload)
    query_tokens = tokenize(query)
    bm25_payload = index_payload.get("movie_bm25", {})
    doc_freq = bm25_payload.get("doc_freq", {})
    doc_count = int(bm25_payload.get("doc_count", len(movies)))
    avg_doc_len = float(bm25_payload.get("avg_doc_len", 1.0))

    sparse_scores = [
        bm25_score(query_tokens, movie.get("term_freq", {}), int(movie.get("doc_len", 0)), doc_freq, doc_count, avg_doc_len)
        for movie in movies
    ]
    dense_scores = [0.0 for _ in movies]
    metadata_boosts: list[float] = []
    reasons_by_movie: list[list[str]] = []

    for movie in movies:
        boost, reasons = movie_metadata_boost(movie, preferences)
        metadata_boosts.append(boost)
        reasons_by_movie.append(reasons)

    embedding_model = index_payload.get("embedding_model") or "text-embedding-3-small"
    resolved_api_key = resolve_api_key_for_model(embedding_model, api_key)
    has_dense = bool(index_payload.get("has_dense_vectors")) and resolved_api_key and index_payload.get("movie_vectors")
    if has_dense:
        try:
            query_vector = embed_query(
                query,
                api_key=resolved_api_key,
                model=embedding_model,
                output_dimensionality=int(index_payload.get("output_dimensionality") or 768),
            )
            matrix = np.array(index_payload["movie_vectors"], dtype=np.float32)
            matrix = np.vstack([normalize_vector(row) for row in matrix])
            dense_scores = list((matrix @ query_vector).astype(float))
        except Exception:
            dense_scores = [0.0 for _ in movies]

    sparse_norm = minmax_scale(sparse_scores)
    dense_norm = minmax_scale(dense_scores)
    metadata_norm = minmax_scale(metadata_boosts)
    generic_hot_query = is_generic_hot_query(preferences)

    ranked: list[dict[str, Any]] = []
    for idx, movie in enumerate(movies):
        if generic_hot_query:
            final_score = 0.80 * metadata_norm[idx] + 0.15 * dense_norm[idx] + 0.05 * sparse_norm[idx]
        else:
            final_score = 0.55 * sparse_norm[idx] + 0.25 * dense_norm[idx] + metadata_boosts[idx]
        if final_score <= 0 and not reasons_by_movie[idx] and sparse_scores[idx] <= 0:
            continue
        ranked.append(
            {
                "score": float(final_score),
                "sparse_score": float(sparse_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "metadata_boost": float(metadata_boosts[idx]),
                "reasons": reasons_by_movie[idx],
                **movie,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    top_items = ranked[:top_n]
    for rank, item in enumerate(top_items, start=1):
        item["rank"] = rank
        item["why_recommended"] = item["reasons"] or ["nội dung và metadata khớp với truy vấn"]
    return top_items


def render_recommendations(
    query: str,
    recommendations: Sequence[dict[str, Any]],
    chat_history: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Biến danh sách recommendation thành câu trả lời text-friendly cho API/UI."""
    if not recommendations:
        return "Mình chưa tìm được phim phù hợp trong dữ liệu hiện có. Bạn có thể thử thêm thể loại, quốc gia, diễn viên hoặc năm phát hành."

    if any(term in normalize_text_for_search(query) for term in ["dang hot", "hot hien nay", "noi bat", "trending", "hot"]):
        lines = ["Một số phim đang hot theo dữ liệu Alpha Cinema:", ""]
    else:
        lines = ["Một số phim phù hợp:", ""]

    for item in recommendations:
        reason = ", ".join(item.get("why_recommended", [])[:2])
        line = f"{item['rank']}. {item.get('title') or item.get('slug')}"
        if item.get("year"):
            line += f" ({item['year']})"
        if reason:
            line += f" - {reason}"
        lines.append(line)
    return "\n".join(lines).strip()


def build_contextual_query_prompt(question: str, chat_history: Sequence[dict[str, Any]] | None) -> str:
    history_block = format_chat_history(chat_history)
    return f"""
Ban la bo tien xu ly query cho RAG Alpha Cinema.

Muc tieu:
- Viet lai tin nhan moi nhat thanh mot cau hoi doc lap de retrieval hieu dung ngu canh.
- Ke thua dung thuc the tu lich su chat nhu ten phim, dien vien, the loai, quoc gia, goi cuoc.
- Neu cau hoi moi da du ro, giu gan nhu nguyen van.
- Chi tra ve 1 dong cau hoi doc lap bang tieng Viet, khong giai thich.

LICH SU CHAT:
{history_block}

TIN NHAN MOI NHAT:
{question}

CAU HOI DOC LAP:
""".strip()


def contextualize_question(
    question: str,
    chat_history: Sequence[dict[str, Any]] | None,
    index_payload: dict[str, Any],
    api_key: str | None = None,
    generation_model: str = "gpt-4o-mini",
) -> str:
    question = normalize_whitespace(question)
    sanitized_history = sanitize_chat_history(chat_history)
    if not sanitized_history or not is_followup_question(question, sanitized_history):
        return question

    rewritten = generate_text_with_model(
        build_contextual_query_prompt(question, sanitized_history),
        api_key=api_key,
        model=generation_model,
        temperature=0.0,
    )
    if rewritten:
        rewritten = normalize_whitespace(rewritten.splitlines()[0])
        if rewritten:
            return rewritten
    return heuristic_contextualize_question(question, sanitized_history, index_payload)


def build_recommendation_prompt(
    question: str,
    standalone_question: str,
    recommendations: Sequence[dict[str, Any]],
    chat_history: Sequence[dict[str, Any]] | None = None,
) -> str:
    history_block = format_chat_history(chat_history)
    recommendation_blocks: list[str] = []
    for item in recommendations[:5]:
        recommendation_blocks.append(
            "\n".join(
                [
                    f"- Tieu de: {item.get('title') or item.get('slug')}",
                    f"  Nam: {item.get('year') or 'khong ro'}",
                    f"  The loai: {', '.join(item.get('categories', [])[:3]) or 'khong ro'}",
                    f"  Quoc gia: {', '.join(item.get('countries', [])[:2]) or 'khong ro'}",
                    f"  Ly do: {', '.join(item.get('why_recommended', [])[:2]) or 'phu hop voi truy van'}",
                ]
            )
        )
    recommendation_text = "\n\n".join(recommendation_blocks)
    return f"""
Ban la tro ly Alpha Cinema.

Quy tac:
- Tra loi tu nhien nhu dang chat tiep noi mot cuoc hoi thoai.
- Chi su dung thong tin trong DANH SACH PHIM.
- Neu day la follow-up, noi tiep mach truoc do, khong lap lai dai dong.
- Neu dang goi y phim, liet ke toi da 5 phim, moi phim 1 dong, kem ly do rat ngan.

LICH SU CHAT:
{history_block}

TIN NHAN MOI NHAT:
{question}

CAU HOI DOC LAP:
{standalone_question}

DANH SACH PHIM:
{recommendation_text}
""".strip()


def build_answer_prompt(
    question: str,
    standalone_question: str,
    retrieved_chunks: Iterable[dict[str, Any]],
    chat_history: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Ghép các chunk đã retrieve thành prompt ràng buộc để model trả lời bám ngữ cảnh."""
    context_blocks: list[str] = []
    for item in retrieved_chunks:
        block = "\n".join(
            [
                f"[{item['chunk_id']}] {item['title']} | section={item['section']} | source={item['source']}",
                item["content"],
            ]
        )
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)
    history_block = format_chat_history(chat_history)
    question_block = question
    if history_block:
        question_block = f"{question}\n\nNgữ cảnh hội thoại gần đây:\n{history_block}"
    if standalone_question and standalone_question != question:
        question_block = f"{question_block}\n\nCâu hỏi độc lập cho truy hồi:\n{standalone_question}"
    question = question_block
    return f"""
Bạn là trợ lý hỗ trợ cho ứng dụng Alpha Cinema.

Quy tắc trả lời:
- Chỉ dùng thông tin trong phần NGỮ CẢNH.
- Nếu ngữ cảnh không đủ, nói rõ: "Mình chưa thấy thông tin này trong dữ liệu Alpha Cinema hiện có.".
- Trả lời bằng tiếng Việt, thật ngắn gọn, tối đa 3 câu.
- Nếu là câu hỏi thao tác, trả lời theo từng bước.
- Nếu là câu hỏi chính sách, ưu tiên nói rõ điều kiện, ngoại lệ và bước tiếp theo.
- Không chèn chunk_id, không thêm dòng nguồn hay trích dẫn kỹ thuật trong câu trả lời.

CÂU HỎI:
{question}

NGỮ CẢNH:
{context}
""".strip()


def _prioritize_primary_chunks(retrieved_chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ưu tiên nguồn primary trước synthetic để fallback/extractive answer đáng tin hơn."""
    primary = [item for item in retrieved_chunks if item.get("metadata", {}).get("source_kind") != "synthetic"]
    synthetic = [item for item in retrieved_chunks if item.get("metadata", {}).get("source_kind") == "synthetic"]
    return primary + synthetic


def strip_source_annotations(text: str) -> str:
    """Xóa phần chú thích nguồn kỹ thuật khỏi answer text trước khi trả về cho UI."""
    if not text:
        return ""
    text = re.sub(r"\n?\s*Nguồn chunk:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n?\s*Source chunk:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return normalize_whitespace(text.replace("\r", "\n"))


def fallback_answer(question: str, retrieved_chunks: Sequence[dict[str, Any]]) -> str:
    """Sinh câu trả lời extractive khi không có LLM hoặc provider generation bị lỗi."""
    if not retrieved_chunks:
        return "Mình chưa thấy thông tin này trong dữ liệu Alpha Cinema hiện có."

    prioritized = _prioritize_primary_chunks(retrieved_chunks)
    top_item = prioritized[0]
    snippet = first_sentences(top_item.get("content", ""), max_sentences=2, max_chars=220)
    if snippet:
        return strip_source_annotations(snippet)
    return strip_source_annotations(str(top_item.get("title") or "Mình chưa thấy thông tin này trong dữ liệu Alpha Cinema hiện có."))


def answer_recommendations(
    question: str,
    standalone_question: str,
    recommendations: Sequence[dict[str, Any]],
    chat_history: Sequence[dict[str, Any]] | None = None,
    api_key: str | None = None,
    generation_model: str = "gpt-4o-mini",
) -> str:
    if not recommendations:
        return render_recommendations(question, recommendations, chat_history=chat_history)

    prompt = build_recommendation_prompt(
        question=question,
        standalone_question=standalone_question,
        recommendations=recommendations,
        chat_history=chat_history,
    )
    generated = generate_text_with_model(
        prompt,
        api_key=api_key,
        model=generation_model,
        temperature=0.25,
    )
    return generated or render_recommendations(question, recommendations, chat_history=chat_history)


def answer_question(
    question: str,
    standalone_question: str,
    retrieved_chunks: Sequence[dict[str, Any]],
    chat_history: Sequence[dict[str, Any]] | None = None,
    api_key: str | None = None,
    generation_model: str = "gpt-4o-mini",
) -> str:
    """Tạo câu trả lời cuối cùng từ chunk đã retrieve, có fallback khi LLM không sẵn sàng."""
    if not retrieved_chunks:
        return "Mình chưa thấy thông tin này trong dữ liệu Alpha Cinema hiện có."

    prioritized_chunks = _prioritize_primary_chunks(retrieved_chunks)
    prompt = build_answer_prompt(
        question=question,
        standalone_question=standalone_question,
        retrieved_chunks=prioritized_chunks,
        chat_history=chat_history,
    )
    generated = generate_text_with_model(
        prompt,
        api_key=api_key,
        model=generation_model,
        temperature=0.2,
    )
    return generated or fallback_answer(standalone_question or question, prioritized_chunks)


def resolve_top_movie_sources(recommendations: Sequence[dict[str, Any]], index_payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """Map recommendation quay lại các chunk nguồn để API vẫn trả được phần sources."""
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in index_payload.get("chunks", [])}
    sources: list[dict[str, Any]] = []
    rank = 1
    for movie in recommendations:
        for chunk_id in movie.get("chunk_ids", [])[:2]:
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            sources.append(
                {
                    "rank": rank,
                    "score": movie.get("score", 0.0),
                    "sparse_score": movie.get("sparse_score", 0.0),
                    "dense_score": movie.get("dense_score", 0.0),
                    "metadata_boost": movie.get("metadata_boost", 0.0),
                    **chunk,
                }
            )
            rank += 1
            if len(sources) >= limit:
                return sources
    return sources


def ask_alpha_cinema(
    question: str,
    index_payload: dict[str, Any],
    api_key: str | None = None,
    top_k: int = 6,
    top_n_recommendations: int = 5,
    generation_model: str = "gpt-4o-mini",
    chat_history: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Entry point chính của hệ thống RAG: detect intent rồi route sang QA hoặc recommendation."""
    sanitized_history = sanitize_chat_history(chat_history)
    standalone_question = contextualize_question(
        question=question,
        chat_history=sanitized_history,
        index_payload=index_payload,
        api_key=api_key,
        generation_model=generation_model,
    )
    intent = detect_intent(standalone_question)
    recommendations: list[dict[str, Any]] = []
    preferences = collect_query_preferences(standalone_question, index_payload)
    requested_count = extract_requested_result_count(standalone_question, top_n_recommendations, max(top_n_recommendations, 10))
    normalized_question = preferences["normalized_query"]
    should_force_recommendation = bool(
        intent == "recommendation"
        or (
            "phim" in normalized_question
            and preferences.get("wants_hot")
            and (
                re.search(r"\b(ke|liet ke|top)\b", normalized_question) is not None
                or "hien nay" in normalized_question
                or "bo phim" in normalized_question
                or "phim nao" in normalized_question
            )
        )
    )

    if should_force_recommendation:
        intent = "recommendation"
        recommendations = recommend_movies(standalone_question, index_payload, api_key=api_key, top_n=requested_count)
        answer = answer_recommendations(
            question=question,
            standalone_question=standalone_question,
            recommendations=recommendations,
            chat_history=sanitized_history,
            api_key=api_key,
            generation_model=generation_model,
        )
        sources = resolve_top_movie_sources(recommendations, index_payload, limit=top_k)
        return {
            "mode": "recommendation",
            "intent": intent,
            "question": question,
            "standalone_question": standalone_question,
            "answer": answer,
            "sources": sources,
            "recommendations": recommendations,
        }

    sources = retrieve(standalone_question, index_payload=index_payload, api_key=api_key, top_k=top_k)

    if intent == "mixed":
        recommendation_query_markers = ["goi y", "de xuat", "nen xem", "tuong tu"]
        if any(marker in normalize_text_for_search(standalone_question) for marker in recommendation_query_markers):
            recommendations = recommend_movies(standalone_question, index_payload, api_key=api_key, top_n=top_n_recommendations)

    answer = answer_question(
        question=question,
        standalone_question=standalone_question,
        retrieved_chunks=sources,
        chat_history=sanitized_history,
        api_key=api_key,
        generation_model=generation_model,
    )
    return {
        "mode": "qa",
        "intent": intent,
        "question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": sources,
        "recommendations": recommendations,
    }
