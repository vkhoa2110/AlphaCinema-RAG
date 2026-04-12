from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


payload = {
    "question": "kể 2 bộ phim đang hot hiện nay",
    "top_k": 6,
    "top_n_recommendations": 5,
}

if len(sys.argv) > 1:
    payload["question"] = " ".join(sys.argv[1:])

request = Request(
    url="http://127.0.0.1:7860/ask",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
        print(body.get("answer", ""))
except HTTPError as exc:
    print(f"HTTP {exc.code}")
    print(exc.read().decode("utf-8", errors="replace"))
except URLError as exc:
    print(f"Request failed: {exc}")
