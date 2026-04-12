---
title: RAG AlphaCinema
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Alpha Cinema RAG Complete

Backend RAG cho ứng dụng **Alpha Cinema**, dùng để:

- hỏi đáp về **phim** từ dữ liệu Firestore
- **gợi ý phim** theo thể loại, quốc gia, diễn viên, năm, độ tuổi
- hỏi đáp về **quy định ứng dụng**: gói cước, thanh toán, hoàn tiền, thiết bị, hồ sơ trẻ em, tài khoản, hỗ trợ

Project này được làm dựa trên:

1. Tài liệu chính sách Alpha (`data/alpha_tai_lieu_rag_dai.md`)
2. Schema collection `movies` trên Firebase mà bạn đã cung cấp
3. Bộ query seed sinh thêm để tăng recall (`data/generated/alpha_seed_queries.md`)

## Kiến trúc

### 1. Nguồn dữ liệu

- **Policy docs**: đọc từ `data/` (`.md`, `.txt`, `.pdf`, `.docx`)
- **Movie data**: export từ Firestore sang `data/firebase_movies_snapshot.json`

### 2. Chunking

#### Policy docs
- tách theo heading Markdown
- chunk theo đoạn
- gắn metadata: `section`, `intent`, `audience`, `source_file`

#### Movie docs
Mỗi phim được tách thành nhiều chunk có cấu trúc:
- `overview`: tên phim, tên gốc, năm, loại, trạng thái, độ tuổi, thể loại, quốc gia
- `cast`: diễn viên, đạo diễn, keyword
- `synopsis_n`: nội dung phim được chunk theo câu

### 3. Retrieval

Hệ thống dùng **hybrid retrieval**:
- **Sparse/BM25** tự cài đặt để luôn chạy được kể cả khi chưa có API key
- **Dense embeddings** bằng OpenAI nếu có `OPENAI_API_KEY`
- **Metadata boosts** cho category, country, actor, director, year, kids-friendly, type, status

### 4. Query routing

Query được phân loại sơ bộ thành:
- `policy`
- `movie`
- `recommendation`
- `mixed`

Ví dụ:
- `Alpha có hoàn tiền khi bị trừ tiền hai lần không?` -> `policy`
- `Phim nào có Châu Vũ Đồng?` -> `movie`
- `Gợi ý phim Trung Quốc thể loại tâm lý` -> `recommendation`

## Cấu trúc thư mục

```text
AlphaCinema_RAG_Complete/
├─ app.py
├─ build_index.py
├─ ingest_firestore.py
├─ rag_core.py
├─ requirements.txt
├─ .env.example
├─ Dockerfile
├─ data/
│  ├─ alpha_tai_lieu_rag_dai.md
│  └─ generated/
│     └─ alpha_seed_queries.md
├─ samples/
│  ├─ firebase_movie_example.json
│  └─ smoke_test_queries.json
└─ storage/
```

## Setup

### 1. Tạo môi trường ảo

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài package

```bash
pip install -r requirements.txt
```

### 3. Tạo file `.env`

```bash
cp .env.example .env
```

Điền các biến cần thiết:

```env
OPENAI_API_KEY=your_key_if_you_want_dense_and_generation
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o-mini
FIREBASE_SERVICE_ACCOUNT_PATH=./secrets/firebase-service-account.json
FIRESTORE_COLLECTION=movies
MOVIES_SNAPSHOT_PATH=./data/firebase_movies_snapshot.json
DATA_DIR=./data
INDEX_PATH=./storage/alpha_cinema_index.json
```

## Bước 1: Export dữ liệu phim từ Firebase

Tạo service account JSON trong Firebase Console rồi lưu tại:

```text
./secrets/firebase-service-account.json
```

Chạy:

```bash
python ingest_firestore.py
```

Lệnh này sẽ đọc collection `movies` và lưu snapshot về:

```text
data/firebase_movies_snapshot.json
```

Nếu bạn chưa có service account ngay, có thể dùng file demo:

```bash
cp samples/firebase_movie_example.json data/firebase_movies_snapshot.json
```

## Bước 2: Build index

### Chế độ sparse-only
Không cần API key, vẫn build và truy hồi được:

```bash
python build_index.py
```

### Chế độ hybrid với OpenAI embeddings
Nếu đã có `OPENAI_API_KEY` trong `.env`, lệnh trên sẽ tự build thêm dense vectors.

## Bước 3: Chạy API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## API

### `GET /`
Health check

### `POST /ask`
Ví dụ request:

```json
{
  "question": "Alpha có hoàn tiền khi bị trừ tiền hai lần không?",
  "top_k": 6,
  "top_n_recommendations": 5
}
```

Ví dụ request khác:

```json
{
  "question": "Phim nào có Châu Vũ Đồng?"
}
```

Ví dụ request gợi ý:

```json
{
  "question": "Gợi ý phim Trung Quốc thể loại tâm lý"
}
```

Ví dụ multi-turn với `session_id` để backend tự nhớ lịch sử chat gần nhất:

```json
{
  "session_id": "demo-user-001",
  "question": "Còn phim Hàn thì sao?",
  "top_k": 6,
  "top_n_recommendations": 5
}
```

Ví dụ gửi thẳng `chat_history` nếu frontend muốn tự kiểm soát memory:

```json
{
  "question": "Nếu qua App Store thì sao?",
  "chat_history": [
    {
      "role": "user",
      "content": "Alpha có hoàn tiền khi bị trừ tiền hai lần không?"
    },
    {
      "role": "assistant",
      "content": "Alpha có xem xét hoàn tiền nếu bạn bị trừ tiền hai lần cho cùng một đơn và cần ảnh sao kê hoặc mã giao dịch để xác minh."
    }
  ]
}
```

### `POST /recommend`
Ví dụ:

```json
{
  "query": "phim gia đình chữa lành",
  "top_n": 5
}
```

### `POST /reload`
Reload index từ file JSON mà không cần restart server.

### `DELETE /sessions/{session_id}`
Xóa memory tạm của một phiên chat để bắt đầu hội thoại mới.

## Ví dụ câu hỏi hệ thống này xử lý tốt

### Về phim
- Phim nào có Châu Vũ Đồng?
- Gợi ý phim Trung Quốc thể loại tâm lý
- Phim 2025 nào có diễn viên Ngô Việt?
- Có phim nào phù hợp cho trẻ em không?
- Phim bộ nào đã hoàn thành?

### Về ứng dụng / chính sách
- Alpha có hoàn tiền không?
- Tôi bị trừ tiền nhưng gói chưa nâng cấp thì làm gì?
- Hủy gói hôm nay có mất quyền xem ngay không?
- Gói Gia Đình xem cùng lúc mấy thiết bị?
- Mua gói qua App Store thì hủy ở đâu?
- Làm sao để chặn trẻ em xem nội dung người lớn?

## Ghi chú triển khai

- Nếu **không có OpenAI API key**, hệ thống vẫn truy hồi bằng BM25 và trả lời ở chế độ fallback extractive.
- Nếu có OpenAI API key, hệ thống sẽ:
  - tạo embeddings cho chunk và movie docs
  - rerank bằng dense similarity
  - sinh câu trả lời tự nhiên hơn
- `data/generated/alpha_seed_queries.md` là dữ liệu sinh thêm để tăng recall, không phải chính sách mới.

## Gợi ý mở rộng tiếp theo

- thêm reranker riêng cho tiếng Việt
- thêm endpoint autocomplete theo title / actor / category
- cache query embeddings
- thêm evaluation script với expected sources
- đồng bộ snapshot Firestore theo lịch bằng cron hoặc CI/CD
