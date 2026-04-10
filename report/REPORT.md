# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Hồ Bảo Thư
**Nhóm:** D4
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* High cosine similarity nghĩa là hai câu hoặc đoạn văn có nội dung, ngữ nghĩa cực kỳ giống nhau, thể hiện qua việc vector biểu diễn của chúng chĩa về cùng một hướng (góc giữa chúng rất nhỏ) trong không gian vector.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi rất thích ăn phở Việt Nam vào buổi sáng."
- Sentence B: "Phở Việt Nam là món điểm tâm sáng mà tôi yêu thích nhất."
- Tại sao tương đồng: Cả hai câu sử dụng cấu trúc khác nhau nhưng mang cùng một thông điệp cốt lõi, nên mô hình biểu diễn sẽ cho ra các vector có cùng hướng.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi rất thích ăn phở Việt Nam vào buổi sáng."
- Sentence B: "Một chiếc máy tính xách tay tốt cần có lượng pin bền bỉ."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn không liên quan (ẩm thực vs công nghệ), do đó chiều vector sẽ khác xa nhau và điểm tương đồng rất thấp.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Cosine similarity chỉ đo lường sự chênh lệch góc (hướng) thay vì khoảng cách tuyệt đối, giúp loại bỏ sự ảnh hưởng khi tài liệu có độ dài khác nhau. Do đó, nó phản ánh đúng sự tương đồng về ngữ nghĩa ổn định hơn Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Step = chunk_size - overlap = 500 - 50 = 450. Số chunk = ceil((Tổng độ dài - overlap) / step) = ceil((10000 - 50) / 450) = ceil(22.11) = 23.
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Khi overlap tăng lên 100 (step giảm còn 400), số lượng chunk sẽ tăng lên thành 25 chunk `ceil((10000-100)/400)`. Tăng overlap giúp đảm bảo thông tin, từ khoá ở ranh giới tách đoạn không bị đứt đoạn, bảo toàn được ngữ cảnh giúp Retrieve ra kết quả chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)
### Domain & Lý Do Chọn

**Domain:** Scientific claim verification / biomedical research abstracts

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì bộ dữ liệu SciFact tập trung vào các claim khoa học và tài liệu nghiên cứu ngắn, rất phù hợp để thử nghiệm bài toán retrieval trong RAG. Dữ liệu có cấu trúc rõ ràng gồm query, document và qrels nên dễ đánh giá mức độ truy xuất đúng/sai. Ngoài ra, domain học thuật giúp nhóm kiểm tra khả năng tìm tài liệu liên quan theo ngữ nghĩa thay vì chỉ khớp từ khóa đơn giản.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Randomized trial of folic acid supplementation and serum homocysteine levels | BeIR/scifact | 1687 | doc_id, title |
| 2 | Keratin-dependent regulation of Aire and gene expression in skin tumor keratinocytes | BeIR/scifact | 1058 | doc_id, title |
| 3 | ALDH1 is a marker of normal and malignant human mammary stem cells and a predictor of poor clinical outcome | BeIR/scifact | 1020 | doc_id, title |
| 4 | Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic | BeIR/scifact | 1990 | doc_id, title |
| 5 | New opportunities: the use of nanotechnologies to manipulate and track stem cells | BeIR/scifact | 640 | doc_id, title |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_id | string | "11705328" | Giúp liên kết document với query và qrels để đánh giá retrieval |
| title | string | "Randomized trial of folic acid..." | Chứa thông tin chính giúp matching nhanh với query |
| source | string | "scifact" | Giúp phân biệt nguồn dữ liệu khi hệ thống mở rộng |
| text_length | integer | 1687 | Hỗ trợ phân tích và tối ưu chunking |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 10 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** __ / __

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
