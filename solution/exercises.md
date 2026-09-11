# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Mỗi mức temperature mình chạy 2 lần (gpt-4o, top_p=0.9). Ở T=0.0 và 0.5, cả hai lần đều kể về hang Sơn Đoòng với cấu trúc gần như giống nhau, chỉ khác vài chi tiết nhỏ. Từ T=1.0 bắt đầu có lần đổi hẳn chủ đề sang cà phê (Việt Nam xuất khẩu cà phê đứng thứ 2 thế giới), cách viết cũng đa dạng hơn, và xuất hiện số liệu lệch như "hang dài khoảng 9 km" so với "hơn 5 km" ở các lần khác. Tóm lại: temperature càng cao thì câu trả lời càng đa dạng và khó đoán, đổi lại độ ổn định và độ chính xác của chi tiết giảm. Ở T=1.5 văn bản chưa bị vỡ, có lẽ vì top_p=0.9 đã cắt bớt những token ít khả năng.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng cần trả lời nhất quán và đúng chính sách: cùng một câu hỏi về đổi trả hay giá cước thì khách nào cũng phải nhận cùng một câu trả lời, và model không được "sáng tác" thêm chi tiết. Thí nghiệm ở Câu 1.1 cho thấy từ T≥1.0 nội dung bắt đầu lệch và sai số liệu. Mình không đặt hẳn 0.0 để câu chữ bớt máy móc khi hội thoại kéo dài. Còn độ chính xác thì nên dựa vào system prompt và tài liệu nội bộ, không trông vào temperature.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Số token đầu ra mỗi ngày: 10.000 × 3 × 350 = 10,5 triệu token (≈ 315 triệu token/tháng).
> - GPT-4o: 10.500K × $0,010 = **$105/ngày** (~$3.150/tháng)
> - GPT-4o-mini: 10.500K × $0,0006 = **$6,30/ngày** (~$189/tháng)
>
> → GPT-4o đắt hơn khoảng **16,7 lần** (0,010 / 0,0006). Giá input cũng chênh đúng tỉ lệ đó (0,0025 / 0,00015), nên tính thêm token đầu vào thì tỉ lệ vẫn giữ ~16,7×. Khi chạy thật `compare_models`, GPT-4o còn chậm hơn: 2,94s so với 1,28s của mini.
>
> **GPT-4o xứng đáng khi:** câu trả lời sai gây thiệt hại lớn và cần suy luận nhiều bước. Ví dụ trợ lý phân tích hợp đồng hoặc tư vấn pháp lý/y tế cho một nhóm nhỏ người dùng: lượng gọi ít, và một lỗi còn tốn kém hơn nhiều so với tiền chênh lệch.
> **Nên dùng mini khi:** lượng gọi lớn và tác vụ đơn giản, như chính kịch bản 10.000 người dùng này nếu là FAQ, phân loại ticket hay tóm tắt ngắn. Với câu hỏi dạng "Việt Nam có bao nhiêu tỉnh?", hai model trả lời gần như y hệt nhau, nên trả gấp 16 lần là phí. Một cách hay là mặc định dùng mini và chỉ chuyển sang GPT-4o khi gặp câu hỏi khó.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, câu trả lời có 129 từ (144 token), gồm một đoạn liền mạch, không dùng thuật ngữ nào và giải thích bằng hình ảnh đời thường: "cuốn sổ lớn ai cũng viết được nhưng không ai xóa được", "ổ khóa" nối các trang, "trao đổi đồ chơi với bạn bè". Với persona chuyên gia tài chính, câu trả lời dài hơn hẳn: chạm trần `max_tokens=256` và bị cắt giữa chừng ("Liên kết kh..."). Câu trả lời này chia thành danh sách có đánh số và dùng dày đặc thuật ngữ như "sổ cái phân tán (DLT)", "hàm băm mật mã (cryptographic hash)", "khối", "dấu vân tay số". Như vậy system prompt không đổi *nội dung kiến thức* mà quyết định *đối tượng người nghe*, từ đó kéo theo độ dài, từ vựng, cách trình bày và kiểu ví dụ. Một hệ quả thực tế: persona "chuyên sâu" tốn gần gấp đôi token đầu ra, nên cần tính cả `max_tokens` khi thiết kế persona.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Mình dùng một đoạn văn tiếng Việt 117 từ về Hà Nội. `count_tokens` (tiktoken, gpt-4o) cho **160 token**, còn ước lượng `117 / 0.75` cho **156**, nên tiktoken **cao hơn khoảng 2,6%**. Với bản dịch tiếng Anh cùng nội dung (109 từ), tiktoken ra 130 token, còn ước lượng ra 145,3, tức ước lượng *thừa* 10,6%. Như vậy công thức "0,75 từ ≈ 1 token" lệch theo hai hướng khác nhau tùy ngôn ngữ. Cùng một nội dung, bản tiếng Việt tốn 160 token so với 130 của tiếng Anh (**nhiều hơn ~23%**), dù ít ký tự hơn (502 so với 579). Nếu dùng bộ mã hóa cũ `cl100k_base` của GPT-4/GPT-3.5, chênh lệch lên tới **265 so với 134 token (~2 lần)**.
>
> Lý do: tokenizer BPE được huấn luyện trên dữ liệu phần lớn là tiếng Anh, nên các từ tiếng Anh thường gộp thành một token nguyên ("people", "known", "elegance"). Tiếng Việt thì có dấu, mỗi chữ có dấu chiếm 2–3 byte UTF-8, và ít cụm ký tự có dấu được học thành token riêng. Vì vậy một âm tiết hay bị tách, ví dụ "Người" → `Ng` + `ười`. Thêm nữa, tiếng Việt là ngôn ngữ đơn lập: mỗi âm tiết viết cách nhau bằng dấu cách, nên "số từ" đếm theo `split()` thực chất là số âm tiết, và một từ như "thanh lịch" đã tốn 2 token. Bộ mã hóa mới `o200k_base` của GPT-4o có từ vựng lớn hơn và nhiều dữ liệu đa ngôn ngữ hơn, nhờ đó thu hẹp khoảng cách này rất nhiều.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi **có người đang ngồi chờ đọc** và câu trả lời dài, như chatbot, trợ lý viết hay giải thích code. Tổng thời gian sinh ra câu trả lời không đổi, nhưng chữ đầu tiên hiện ra gần như ngay lập tức nên người dùng *cảm thấy* nhanh, đọc được đến đâu hay đến đó, và dừng được sớm nếu model đi sai hướng. Ví dụ ở Câu 2.1, lời gọi non-streaming với persona giáo viên mất 6,14 giây, và suốt thời gian đó màn hình trống trơn. Ngược lại, non-streaming phù hợp hơn khi **không có người đọc từng chữ** hoặc cần xử lý trọn vẹn kết quả trước khi dùng: pipeline chạy nền (batch tóm tắt, phân loại ticket), đầu ra phải parse thành JSON hay gọi tool, cần kiểm duyệt nội dung *trước* khi hiển thị, hoặc cần đọc `usage` để tính chi phí chính xác. Những trường hợp này cần cả khối kết quả một lần, streaming chỉ làm code phức tạp hơn (ghép chunk, xử lý chunk `None`, lỗi giữa chừng) mà không đem lại lợi ích gì.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Khi API quá tải, server cần *thời gian* để hồi phục. Delay cố định (luôn chờ 1 giây) cứ đều đặn bắn yêu cầu vào server đang nghẹt, nên áp lực không bao giờ giảm. Exponential backoff (0,1s → 0,2s → 0,4s → 0,8s…) thì tự giãn tần suất thử lại càng lúc càng thưa. Lỗi thoáng qua vẫn được retry nhanh ở lần đầu, còn khi sự cố kéo dài thì client tự "lùi" ra, nhường chỗ cho server phục hồi và giảm số lần gọi phí công (cũng như rate-limit bị đốt). Nếu hàng nghìn client cùng retry với delay cố định giống nhau, chúng sẽ **thử lại đồng loạt tại cùng một thời điểm**, tạo thành những đợt sóng request dồn dập (hiệu ứng *thundering herd*). Server vừa hồi phục chút ít lại bị đánh sập ngay, và cả hệ thống kẹt trong vòng lặp quá tải → lỗi → retry đồng loạt → quá tải. Vì vậy trong thực tế người ta còn cộng thêm **jitter** (một khoảng ngẫu nhiên, ví dụ `random.uniform(0, delay)`) để các client lệch pha nhau, và đặt trần delay tối đa. Hàm `retry_with_backoff` hiện tại chưa có jitter, đây là điểm nên bổ sung khi dùng thật.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Mình chọn persona **trợ giảng của khóa AI**, dùng trong `python template.py`:
>
> *"Bạn là trợ giảng thân thiện của khóa AI Practical Competency. Luôn trả lời bằng tiếng Việt, ngắn gọn tối đa 3 câu. Nếu câu hỏi nằm ngoài chủ đề AI/lập trình, lịch sự từ chối và gợi ý quay lại bài học. Nếu không chắc chắn, hãy nói rõ là không chắc thay vì bịa."*
>
> - **"Luôn trả lời bằng tiếng Việt"**: câu hỏi của học viên thường lẫn thuật ngữ tiếng Anh ("temperature", "top_p", "streaming"), nên nếu không chỉ định thì model dễ trả lời bằng tiếng Anh. Chữ "luôn" giữ ngôn ngữ ổn định suốt phiên. Tiếng Việt cũng tốn token hơn (Câu 2.2), nên đây là lựa chọn có chủ đích chứ không phải mặc định.
> - **"Ngắn gọn tối đa 3 câu"**: dùng con số cụ thể thay vì chỉ nói "ngắn gọn" chung chung, vì model hiểu "ngắn gọn" rất co giãn. Lợi ích là đọc dễ trong terminal, giảm token đầu ra, tức giảm chi phí và độ trễ. Ngoài ra history được giữ 3 lượt, nên câu trả lời ngắn thì mỗi lượt sau tốn ít token đầu vào hơn. Khi chạy thật 5 lượt, mọi câu trả lời đều ≤ 3 câu, cả phiên chỉ 284 token.
> - **Luật từ chối ngoài chủ đề** giúp trợ lý không bị dùng vào việc khác: khi hỏi "công thức nấu phở bò", nó lịch sự từ chối và mời quay lại bài học. Còn câu **"không chắc thì nói không chắc"** nhằm giảm bịa, vì ở Block 1 cả hai model đều tự tin trả lời "63 tỉnh thành", một con số đã cũ.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất: history chỉ giữ 3 lượt, phần cũ bị cắt là mất hẳn.** Mình thấy rõ điều này khi chạy thử 5 lượt. Ở lượt 4, hỏi "Câu hỏi đầu tiên mình hỏi là gì?" thì trợ lý vẫn trả lời đúng, vì lượt 1 còn trong history. Nhưng sang lượt 5, lượt 1 đã bị cắt. Khi được nhờ "tóm tắt 2 ý chính bạn vừa nói về temperature", nó chỉ tóm được ý về mức 0.2–0.5 cho CSKH ở lượt 2, rồi tách chính ý đó thành 2 gạch đầu dòng. Định nghĩa temperature ở lượt 1 hoàn toàn biến mất mà trợ lý không hề hay biết. Một hạn chế phụ: `total_cost` chỉ tính token của `user_msg` + `reply`, trong khi mỗi lượt thật sự gửi lên cả system prompt và history. Vì vậy chi phí báo cáo thấp hơn thực tế.
>
> **Cải thiện đề xuất: bộ nhớ tóm tắt cuộn (rolling summary).** Thay vì vứt các message bị cắt, ta nén chúng thành một bản tóm tắt:
> 1. Giữ thêm biến `summary = ""`. Trước khi làm `history = history[-6:]`, lấy phần sắp bị cắt `dropped = history[:-6]`.
> 2. Nếu `dropped` khác rỗng, gọi `chat_with_system_prompt` với **gpt-4o-mini** (rẻ hơn ~16 lần, Câu 1.3): *"Cập nhật bản tóm tắt hội thoại sau, tối đa 5 gạch đầu dòng, giữ các dữ kiện người dùng đã nêu"*, truyền vào `summary` cũ + `dropped`, rồi bọc trong `retry_with_backoff`.
> 3. Khi ghép messages, gắn tóm tắt vào system prompt: `{"role": "system", "content": persona + "\n\nTóm tắt hội thoại trước đó:\n" + summary}`.
> 4. Nâng cấp thêm: cắt history theo **ngân sách token** (dùng `count_tokens`, ví dụ ≤ 1.500 token) thay vì cố định 6 message. Đồng thời sửa thống kê để `estimate_cost` tính trên toàn bộ `messages` gửi đi, cho số liệu chi phí khớp hóa đơn thật.
>
> Như vậy trợ lý nhớ được ý chính của cả phiên, mà token đầu vào mỗi lượt vẫn bị chặn trần.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
