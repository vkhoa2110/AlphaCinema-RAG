from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_core import ask_alpha_cinema, load_index


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
# Smoke test đơn giản: nạp index mẫu, chạy các query mẫu và in kết quả để kiểm tra pipeline đầu cuối.
index = load_index(ROOT / 'storage' / 'alpha_cinema_index.json')
queries = json.loads((ROOT / 'samples' / 'smoke_test_queries.json').read_text(encoding='utf-8'))

for question in queries:
    result = ask_alpha_cinema(question, index_payload=index, api_key=None)
    print('=' * 80)
    print('Q:', question)
    print('Intent:', result['intent'], '| Mode:', result['mode'])
    print('Answer:\n', result['answer'])
    print('Top sources:', [item['chunk_id'] for item in result['sources'][:3]])
    if result.get('recommendations'):
        print('Recommendations:', [item['title'] for item in result['recommendations']])

print('=' * 80)
print('Multi-turn conversation check')
history: list[dict[str, str]] = []
conversation = [
    'Gợi ý phim Trung Quốc thể loại tâm lý',
    'Còn phim Hàn thì sao?',
]

for question in conversation:
    result = ask_alpha_cinema(
        question,
        index_payload=index,
        api_key=None,
        chat_history=history,
    )
    print('-' * 80)
    print('Q:', question)
    print('Standalone:', result.get('standalone_question'))
    print('A:', result['answer'])
    history.extend(
        [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': result['answer']},
        ]
    )
