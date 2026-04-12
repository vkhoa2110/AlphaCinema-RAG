from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_core import RAGError, build_index_payload, save_index


ROOT = Path(__file__).resolve().parent
load_dotenv()


def resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    """Đổi relative path sang absolute path theo thư mục gốc của project."""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def main() -> None:
    """CLI build toàn bộ RAG index từ tài liệu policy và snapshot phim."""
    parser = argparse.ArgumentParser(description="Build Alpha Cinema RAG index")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "./data"),
        help="Directory containing policy and support documents",
    )
    parser.add_argument(
        "--movies-snapshot",
        default=os.getenv("MOVIES_SNAPSHOT_PATH", "./data/firebase_movies_snapshot.json"),
        help="JSON snapshot exported from Firestore",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("INDEX_PATH", "./storage/alpha_cinema_index.json"),
        help="Where to save the generated index",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model to use for dense vector generation",
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=int(os.getenv("OUTPUT_DIMENSIONALITY", "768")),
        help="Embedding output dimensionality",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        help="Number of texts to send per embedding request",
    )
    parser.add_argument(
        "--embedding-requests-per-minute",
        type=int,
        default=int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "90")),
        help="Throttle embedding requests to stay under provider rate limits; set 0 to disable",
    )
    parser.add_argument(
        "--embedding-max-retries",
        type=int,
        default=int(os.getenv("EMBEDDING_MAX_RETRIES", "6")),
        help="How many times to retry an embedding batch after temporary provider quota errors",
    )
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir, ROOT)
    movies_snapshot = resolve_path(args.movies_snapshot, ROOT)
    output_path = resolve_path(args.output, ROOT)

    index_payload = build_index_payload(
        data_dir=data_dir,
        movies_snapshot_path=movies_snapshot,
        api_key=None,
        embedding_model=args.embedding_model,
        output_dimensionality=args.output_dimensionality,
        embedding_batch_size=args.embedding_batch_size,
        embedding_requests_per_minute=args.embedding_requests_per_minute,
        embedding_max_retries=args.embedding_max_retries,
    )
    saved_path = save_index(index_payload=index_payload, output_path=output_path)
    print(
        "Built Alpha Cinema index from "
        f"{index_payload['file_count']} policy files, "
        f"{index_payload['movie_count']} movies, "
        f"{index_payload['chunk_count']} chunks -> {saved_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except RAGError as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
