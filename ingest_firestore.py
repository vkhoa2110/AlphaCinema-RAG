from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_core import RAGError, fetch_movies_from_firestore, json_dumps


ROOT = Path(__file__).resolve().parent
load_dotenv()


def resolve_path(path_value: str, base_dir: Path) -> Path:
    """Chuẩn hóa đường dẫn đầu vào để script chạy ổn từ nhiều vị trí khác nhau."""
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def main() -> None:
    """CLI export collection phim từ Firestore ra JSON snapshot dùng cho bước build index."""
    parser = argparse.ArgumentParser(description="Export movies from Firebase Firestore")
    parser.add_argument(
        "--service-account",
        default=os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "./secrets/firebase-service-account.json"),
        help="Path to Firebase service account JSON",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("FIRESTORE_COLLECTION", "movies"),
        help="Firestore collection name",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MOVIES_SNAPSHOT_PATH", "./data/firebase_movies_snapshot.json"),
        help="Where to save the JSON snapshot",
    )
    args = parser.parse_args()

    service_account_path = resolve_path(args.service_account, ROOT)
    output_path = resolve_path(args.output, ROOT)

    movies = fetch_movies_from_firestore(service_account_path, collection_name=args.collection)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dumps(movies), encoding="utf-8")
    print(f"Exported {len(movies)} movies -> {output_path}")


if __name__ == "__main__":
    try:
        main()
    except RAGError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
