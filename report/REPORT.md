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
| Folic acid... | FixedSizeChunker (`fixed_size`) | 4 | ~421.0 | Không tốt (cắt ngang câu) |
| Folic acid... | SentenceChunker (`by_sentences`) | 5 | ~337.0 | Tốt (ngữ nghĩa trọn câu) |
| Folic acid... | RecursiveChunker (`recursive`) | 4 | ~400.0 | Rất tốt (theo đoạn/câu) |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> Strategy này chia nhỏ văn bản dựa trên danh sách các dấu phân cách ưu tiên từ lớn đến nhỏ (ví dụ: `\n\n`, `\n`, `. `, v.v.). Nó đệ quy cắt văn bản cho tới khi mỗi phần (chunk) đạt dưới giới hạn `chunk_size`. Cuối cùng, các phần quá ngắn sẽ được gộp lại tối ưu mà không bị chia cắt vô lý.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain tóm tắt luận văn y khoa (biomedical abstracts) thường có cấu trúc theo từng đoạn hoặc các câu ghép phức tạp. RecursiveChunker giúp tôn trọng ranh giới câu và đoạn văn bản, do đó giữ trọn vẹn được context mang tính chuyên ngành mà không bị cắt ngẫu nhiên.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Folic acid... | best baseline (Sentence) | 5 | ~337 | Khá tốt |
| Folic acid... | **của tôi** (Recursive) | 4 | ~400 | Tốt, đầy đủ context hơn |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive | 9.0 | Chunk gọn, trọn khối ý | Code phức tạp |
| Bạn Tuấn | Fixed Size | 6.5 | Triển khai nhanh | Chặt đứt câu |
| Bạn Việt | Sentence | 8.0 | Lấy full câu đúng ý | Chunk size không đều nhau |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Recursive Strategy là tốt nhất. Do domain học thuật chứa rất nhiều câu văn và đoạn văn dài ngắn bất định. Recursive cho phép chunking linh hoạt với các đoạn xuống dòng, vừa giúp giới hạn độ dài chunk, vừa không chặt chém ngữ nghĩa ở giữa câu.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng thư viện `re` với biểu thức chính quy (regex) `\. |\! |\? |\.\n` để phát hiện dấu kết thúc câu, sau đó nhóm các câu liên tiếp lại tới giới hạn `max_sentences_per_chunk`. Các khoảng trắng thừa được loại bỏ bằng `.strip()`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Hàm đệ quy kiểm tra xem đoạn text hiện tại đã nhỏ hơn chunk_size hay chưa. Nếu lớn hơn, sử dụng ký tự phân cách (separator) mạnh nhất (như `\n\n`) để cắt, rồi đệ quy chia nhỏ các phần bằng các separator yếu hơn. Cuối cùng thực hiện merge các đoạn nhỏ lại.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Documents được lưu vào một dictionaries (tra cứu nhanh) và một list. Vector embeddings tính qua `embedding_fn`. Khi search, dùng `compute_similarity` dựa theo góc vector quét qua tập embeddings và sắp xếp giảm dần bằng `sorted(_, reverse=True)` để lấy ra `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> Filter trước (pre-filtering) bằng cách duyệt qua `metadata_filter` loại bỏ documents không khớp trước khi tính similarity, giúp tiết kiệm tính toán. Delete được thực hiện bằng vòng lặp và điều kiện xoá index ra khỏi các collection của store.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search()` để lấy `top_k` chunks phù hợp nhất với query. Gói gọn context này vào prompt "Answer based on context" rồi truyền gọi hàm LLM để sinh kết quả hoàn chỉnh dựa theo context truy cập được từ vector store.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 1.33s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Machine learning is widely used." | "AI applications are very popular." | high | 0.81 | Yes |
| 2 | "An apple a day keeps the doctor away." | "Doctors recommend eating fruits daily." | high | 0.84 | Yes |
| 3 | "The cat is sleeping on the sofa." | "Quantum mechanics deals with physics." | low | -0.12 | Yes |
| 4 | "Deep learning uses neural networks." | "Neural networks are core to deep learning." | high | 0.95 | Yes |
| 5 | "I like to eat pizza." | "Cars run on gasoline." | low | 0.05 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Pair 2: "An apple a day..." và "Doctors recommend...". Dù khác biệt về mặt cấu trúc ngữ pháp và số lượng từ vựng trùng lập, vector nhúng vẫn đánh giá score tương đồng cao do cùng mang ý niệm về sức khỏe, y tế. Nhờ vậy, thấy được embedding thực chất biểu diễn trên không gian hướng ngữ nghĩa (Semantic) chứ không phải khớp từ (Keyword matching) đơn thuần.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is the effect of folic acid supplementation on serum homocysteine? | It reduces serum homocysteine levels significantly. |
| 2 | What regulates Aire expression in skin tumor keratinocytes? | Aire is transcriptionally regulated by Keratin-dependent signaling pathways. |
| 3 | Is ALDH1 an effective predictive marker for human mammary stem cells? | Yes, ALDH1 is an effective prognostic marker for malignant human mammary stem cells. |
| 4 | Is variant prion protein prevalent in human appendixes after BSE exposure? | Yes, abnormal prion protein was found to be highly prevalent in appendix survey. |
| 5 | How are nanotechnologies used with stem cells? | Nanotechnologies help isolate, manipulate, and track stem cell activities safely. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What is the effect of folic acid?| ... folic acid supplementation lowered serum homocysteine... | 0.85 | Yes | Folic acid lowers serum homocysteine significantly. |
| 2 | What regulates Aire expression? | ... keratinocyte turnover controls Aire gene regulation...   | 0.82 | Yes | It's controlled by keratinocyte turnover processes. |
| 3 | Is ALDH1 a marker?             | ... ALDH1 serves as a marker for normal and malignant stem...| 0.89 | Yes | Yes, ALDH1 serves as an effective tumor stem marker. |
| 4 | Prevalence of prion protein?   | ... survey shows prevalent abnormal prion protein...         | 0.90 | Yes | Prevalent in human appendixes following exposure. |
| 5 | Nanotech and stem cells?       | ... applications of nanotechnology to track stem cell...     | 0.86 | Yes | Used to track cell activity and manipulate differentiation.|

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nhóm đã chỉ ra cách thiết kế Metadata một cách khoa học để `search_with_filter` hoạt động cực kỳ hiệu quả. Chẳng hạn, thêm tag metadata nhỏ thay vì parse text giúp giảm hẳn thời gian compute Similarity.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Tôi thấy họ sử dụng Hybrid Search (kết hợp TF-IDF/BM25 cùng Vector Embeddings) để bù đắp các lỗ hổng của vector khi phải search chính xác các keyword hiếm, viết tắt hoặc thông số kỹ thuật.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ code thêm bộ tiền xử lý loại bỏ hoàn toàn các ký tự Markdown hay HTML thừa ở từng Document trước khi truyền vào Chunker. Định dạng tài liệu lộn xộn khiến Chunker cắt bị sai ngữ cảnh thay vì cắt ở dấu kết thúc câu.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 0 / 5 |
| **Tổng** | | **95 / 100** |
