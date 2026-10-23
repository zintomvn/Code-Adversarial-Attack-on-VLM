# Báo cáo kỹ thuật thực nghiệm H1: Counterfactual negatives, attention và MPC

**Ngày lập:** 17/10/2026.  
**Code được đối chiếu:** [kltn-attack-v4.3.3-h1-cfvlm.ipynb](kltn-attack-v4.3.3-h1-cfvlm.ipynb).  
**Protocol đánh giá:** [text_gen_eval_v3.1.ipynb](text_gen_eval_v3.1.ipynb).  
**SHA-256 notebook attack tại thời điểm lập báo cáo:** 3c17d4fb5667292a3d757d9d798fdd7338f3f6a81769e1ad4fe58fcb2056c86e.

Đây là báo cáo về thiết kế, cơ chế thực thi và cách diễn giải thực nghiệm. Các kỳ vọng dưới đây là giả thuyết cần kiểm tra, không phải kết luận đã được chứng minh bằng kết quả. Tên file dùng H1 theo yêu cầu; nội dung bao gồm cả H1–H4 được khai báo trong MOLAB_PLAN.

## 1. Mục tiêu và phạm vi

Thực nghiệm kiểm tra khả năng tạo nhiễu trên ảnh source để một VLM đích sinh mô tả gần với mô tả của ảnh target. Gradient được lấy từ các surrogate trong quá trình attack; hai victim đánh giá không được dùng để tính gradient attack.

Chiến dịch hiện tại có **8 cấu hình × 200 cặp ảnh × 300 bước**, tương đương **1.600 lượt tối ưu ảnh và 480.000 bước**, chưa tính smoke test, captioning, sinh negatives và GPT judging. Tám cấu hình tạo ra **6 phép so sánh giả thuyết chính**; không phải 8 giả thuyết độc lập.

FOA reference và MPC-CLIP là các baseline đã được chạy ở giai đoạn trước, nằm ngoài tám cấu hình này. Kết quả cũ chỉ được đưa vào bảng so sánh nếu xác minh tương thích dữ liệu, ngân sách, model revisions, preprocessing và protocol evaluation. MPC-CLIP không đại diện cho full MPC có thêm InternVL/DINOv2.

## 2. Thiết lập chung

| Hạng mục                 | Thiết lập                                                         |
| ------------------------ | ----------------------------------------------------------------- |
| Dữ liệu                  | 200 cặp source–target có thứ tự cố định, manifest pairs_200.jsonl |
| Kiểm tra dữ liệu         | Hash ảnh và manifest; không tự thay cohort hoặc bỏ mẫu lỗi        |
| Seed                     | 2023                                                              |
| Kích thước attack        | 224 × 224                                                         |
| Số bước mỗi ảnh          | 300                                                               |
| Ngân sách nhiễu          | L-infinity không quá 16/255 trên ảnh chuẩn hóa                    |
| Bước cập nhật            | 1/255 trên ảnh chuẩn hóa                                          |
| Đơn vị nội bộ            | Pixel trong thang 0–255; epsilon = 16, alpha = 1                  |
| Ensemble surrogate       | CLIP B16, CLIP B32, Laion CLIP ViT-G/14                           |
| Precision hybrid         | FP32; HYBRID_AMP = False                                          |
| Gradient engine          | full_backward_serial_engine; chỉ delta là leaf được tối ưu        |
| FOA local clusters       | 10                                                                |
| Random crop              | Scale 0,5–0,9; dùng cho source và target theo cấu hình chung      |
| Hệ số objective mặc định | FOA = 1; MPC = 1; negative loss = 0,5                             |
| MPC fusion mặc định      | Lambda = 0,6                                                      |
| Negative bank            | 4 caption counterfactual đã qua audit cho mỗi cặp                 |
| Negative được chọn       | K_NEG = 2                                                         |
| Negative mining          | Mỗi 10 bước                                                       |
| Attention                | Qwen3-VL-8B-Instruct, layer 29, input 448                         |
| Attention refresh        | Mỗi 30 bước cho dynamic và random-region                          |
| Region schedule          | 10 mức diện tích, tối đa 3 box/mức, IoU cap 0,3 và fallback       |
| Momentum hybrid          | 0                                                                 |
| Checkpoint ảnh cục bộ    | Mỗi 30 bước                                                       |
| Resume attack            | Ranh giới ảnh hoàn tất; khôi phục FOA history và RNG              |

Model revisions được khóa bằng commit SHA trong plan/run_config. Cùng caption factual và cùng bank đã audit được dùng cho các variant liên quan. FOA giữ trạng thái giữa các ảnh nên thứ tự cohort là một phần của protocol.

## 3. Objective và vòng lặp attack

### 3.1. Objective tổng

Gọi ảnh adversarial ở bước t là:

$$
x_a^{(t)} = x_s + \delta^{(t)}.
$$

Hybrid tối đa hóa:

$$
J = w_F S_{\mathrm{FOA}} + w_M S_{\mathrm{MPC}} - w_N L_{\mathrm{neg}}.
$$

Mô hình được đóng băng. Mỗi bước lấy gradient theo delta, cập nhật theo dấu gradient và chiếu delta về miền ngân sách:

$$
\delta^{(t+1)}
= \operatorname{clip}\left(
\delta^{(t)}+\alpha\,\operatorname{sign}(g^{(t)}),
-\epsilon,+\epsilon\right).
$$

Với hybrid hiện tại, momentum bằng 0 nên hướng cập nhật là gradient của bước hiện tại. Code không clamp toàn bộ ảnh adversarial sau mọi bước theo một protocol mới; clamp/preprocessing của encoder và clamp ảnh xuất cuối giữ theo implementation hiện có.

### 3.2. FOA

FOA trích đặc trưng global và local qua ba backbone. Đặc trưng local được phân cụm K-means, sau đó dùng cơ chế Optimal Transport và trọng số ensemble của implementation FOA được nhúng trong notebook.

Ở mỗi bước có crop mới của target. Khi source crop được bật, code lần lượt:

1. Tính FOA trên ảnh adversarial đầy đủ để cập nhật lịch sử trọng số.
2. Tính FOA trên crop của ảnh adversarial để đóng góp vào objective và gradient.

Lượt ảnh đầy đủ được chạy không tạo graph gradient, nhưng vẫn cần cho lịch sử FOA. Bỏ lượt này hoặc cache target crop xuyên suốt 300 bước sẽ thay đổi thuật toán. Lịch sử FOA không được reset tùy tiện giữa hai ảnh.

### 3.3. MPC-CLIP và image–text fusion

Với từng backbone, code trộn embedding ảnh và caption factual:

$$
a_s = \operatorname{concat}_b[
\lambda z_{s,b} + (1-\lambda)t_{s,b}],\qquad
a_t = \operatorname{concat}_b[
\lambda z_{t,b} + (1-\lambda)t_{t,b}].
$$

Đặc trưng ảnh adversarial được concatenate theo cùng thứ tự backbone. Các vector concatenate được chuẩn hóa trước khi tính MPC score:

$$
S_{\mathrm{MPC}}
= -\operatorname{CE}\left(
\left[
\frac{\omega\cos(z_a,a_t)}{\tau},
\frac{\cos(z_a,a_s)}{\tau}
\right],\,0\right),
\qquad \omega=2,\ \tau=0{,}2.
$$

Tối đa hóa score này khuyến khích nghiêng về target thay vì source. Đây là CE trên embedding concatenate, không phải trung bình ba CE độc lập. Anchors được chuẩn bị theo ảnh; đặc trưng adversarial thay đổi theo mỗi bước.

### 3.4. Negative loss

Positive text là caption factual của target. Negatives là các caption counterfactual của chính target, không phải caption của ảnh khác được lấy ngẫu nhiên từ dataset.

Với từng backbone, đặt:

$$
s^+=\cos(z_a,t_t),\qquad s_i^-=\cos(z_a,n_i).
$$

Loss mặc định:

$$
L_{\mathrm{neg}}
=\frac{1}{B}\sum_{b=1}^{B}
\log\left(1+\sum_{i\in\mathcal{N}}
\exp\frac{s^-_{b,i}-s^+_b}{\tau_n}\right),
\qquad B=3,\ \tau_n=0{,}2.
$$

Hybrid trừ loss này khỏi objective. Vì vậy, nó khuyến khích phân biệt target text với các mô tả counterfactual gần nghĩa, thay vì chỉ tăng similarity với target.

Hard mining chấm bank trên **ảnh adversarial đầy đủ**, rồi lấy hai caption có mean cosine cao nhất qua ba CLIP. Negative loss trong objective lại dùng **crop hiện tại**. Hai lượt này không thể coi là phép tính trùng hoàn toàn vì đầu vào ảnh khác nhau.

### 3.5. Một lượt tối ưu ảnh

```text
Đọc source/target, caption và bank đúng hash/policy
Khởi tạo delta; chuẩn bị anchors cần thiết
Lặp t = 0..299:
    Nếu có region và đến kỳ refresh: tính attention và schedule
    Chọn entry của schedule theo bước tuyệt đối
    Nếu có negatives và t chia hết cho 10: chọn hai negative
    Tạo target crop và source/adversarial crop
    Tính FOA, MPC và negative loss theo variant
    Backward theo delta; kiểm tra gradient hữu hạn
    Cập nhật delta trong ngân sách
    Ghi trace; mỗi 30 bước lưu ảnh checkpoint
Nếu có region: tính thêm map ở bước 300 để chẩn đoán
Xuất ảnh, state/RNG, complete record và manifest
Sao lưu và xác minh checkpoint trên Drive
```

## 4. Chính sách caption negative theo CF-VLM

CF-VLM phân biệt can thiệp vào một thuộc tính đối tượng và can thiệp vào quan hệ nhân quả. Thực nghiệm này mượn nguyên tắc chỉnh sửa counterfactual có kiểm soát để xây dựng caption negatives. Nó không tái tạo quy trình sinh ảnh counterfactual hoặc fine-tuning của paper. Xem [CF-VLM, §2.2](https://arxiv.org/html/2506.17267v1#S2.SS2) và [repository chính thức](https://github.com/HCP-AI-Research-Lab/CF-VLM).

Implementation dùng policy **cfvlm_text_edits_v2_audited**:

- Chỉnh đúng một thuộc tính của một đối tượng; hoặc chỉnh một liên kết cause–effect có sẵn, kể cả kết quả phụ thuộc cần thay theo.
- Giữ nguyên các thông tin không liên quan.
- Thông tin được sửa phải mâu thuẫn với ảnh factual; thêm chi tiết không quan sát được không đủ điều kiện.
- Kiểm tra JSON, vị trí span, chỉnh sửa không chồng lấn và caption được tái dựng đúng.
- Qwen nhận target image + factual caption để sinh ứng viên; một lượt Qwen khác audit tính tối thiểu, giữ ngữ cảnh, mâu thuẫn hình ảnh, tính mạch lạc và nhãn can thiệp.
- Chỉ giữ đủ bốn ứng viên được chấp nhận; có retry và phản hồi lý do bị loại, không dùng template fallback hoặc tự bỏ ảnh thiếu bank.

Ví dụ minh họa: “A red car waits beside a tree” đổi red thành blue là một can thiệp màu nếu ảnh thực sự cho thấy xe đỏ. Đổi đồng thời màu xe và loại cây là hai can thiệp độc lập.

Audit do cùng một Qwen thực hiện nên vẫn có sai số và thiên lệch chung. Cần xem đây là bộ lọc tự động, không phải chứng nhận chất lượng bằng con người. Không dùng lại bank cũ chỉ đổi danh từ nhưng không qua policy hiện tại.

Một số Markdown mô tả lịch sử trong notebook còn nhắc “subject/object phrase”. Khi tái lập v4.3.3 phải dùng các hàm CF validation và policy trên làm nguồn chuẩn cho hành vi thực thi.

## 5. Attention và lịch vùng

Attention backend teacher-force **source caption cố định** vào assistant suffix của Qwen, lấy các query rows của content tokens tới image-token keys ở layer 29. Softmax được tính trên toàn bộ causal keys trước khi chọn image keys, sau đó tổng hợp theo head/token.

“Causal” ở đây mô tả decoder causal self-attention. Nó không chứng minh vùng có vai trò nhân quả đối với quyết định của victim.

Map được chuyển thành chuỗi box theo 10 mức diện tích từ 0,1 tới 1,0. Source crop được lấy theo box của entry hiện tại; gradient đi qua crop primitive. Code không bổ sung một pixel mask độc lập hoặc cơ chế temporal-credit/EMA.

Dynamic refresh ở các bước 0, 30, …, 270. Sau refresh, entry vẫn được chọn theo **bước tuyệt đối** trong 300 bước, không quay lại phase đầu.

Random-region giữ schedule/diện tích/hình dạng do map tạo ra rồi ngẫu nhiên hóa vị trí box. Vì vậy, đây là đối chứng **vị trí attention** trong cơ chế vùng hiện tại; chưa phải đối chứng vùng ngẫu nhiên hoàn toàn độc lập với attention.

## 6. Ma trận tám cấu hình

| ID / tên đầy đủ                          | Objective             | Negative selection | Region                | Attention tối ưu / chẩn đoán mỗi ảnh |
| ---------------------------------------- | --------------------- | ------------------ | --------------------- | -----------------------------------: |
| primary_ablations / hybrid_plain         | FOA + MPC             | Không              | Không                 |                                0 / 0 |
| primary_ablations / hybrid_negative      | FOA + MPC − 0,5 Lneg  | Hard top-2         | Không                 |                                0 / 0 |
| primary_ablations / hybrid_fixed         | FOA + MPC − 0,5 Lneg  | Hard top-2         | Map cố định           |                                1 / 1 |
| primary_ablations / hybrid_dynamic       | FOA + MPC − 0,5 Lneg  | Hard top-2         | Refresh 30 bước       |                               10 / 1 |
| primary_ablations / hybrid_random_region | FOA + MPC − 0,5 Lneg  | Hard top-2         | Vị trí box ngẫu nhiên |                               10 / 1 |
| primary_ablations / hybrid_random_neg    | FOA + MPC − 0,5 Lneg  | Random-2           | Dynamic như trên      |                               10 / 1 |
| h4_mpc_zero / hybrid_plain               | FOA; trọng số MPC = 0 | Không              | Không                 |                                0 / 0 |
| h4_image_only / hybrid_plain             | FOA + MPC; lambda = 1 | Không              | Không                 |                                0 / 0 |

Hai dòng H4 đều có variant tên hybrid_plain nhưng thuộc **job khác nhau**. Khi đọc kết quả phải dùng job + variant, không chỉ dùng tên thư mục variant.

### 6.1. hybrid_plain — mốc hybrid không có negative/region

**Cách hoạt động:** kết hợp FOA và MPC-CLIP trên crop thông thường. MPC dùng anchors trộn ảnh–text với lambda = 0,6. Không sinh bank phục vụ variant này và không tính Qwen attention trong vòng tối ưu.

**Kiểm tra điều gì:** làm mốc đối chứng cho việc thêm negative loss; đồng thời là treatment của hai phép so sánh H4.

**Giới hạn:** chạy cấu hình này thành công không tự chứng minh hybrid tốt hơn FOA hoặc full MPC. Cần kết quả baseline tương thích để trả lời câu hỏi đó.

### 6.2. hybrid_negative — H1: đóng góp của negative loss

**Cách hoạt động:** giữ FOA + MPC và random crop như hybrid_plain, thêm bank CF-VLM đã audit. Mỗi 10 bước lấy hard top-2 và thêm negative loss vào objective với dấu trừ.

**So sánh chính:** hybrid_negative − hybrid_plain.

**Kỳ vọng:** ASR/AvgSim trên victim tăng khi có mục tiêu phân biệt target với counterfactual.

**Kết luận được phép:** chênh lệch hỗ trợ hoặc không hỗ trợ lợi ích của **gói negative loss + hard selection** trong thiết lập này. So sánh này chưa cô lập riêng chất lượng prompt CF-VLM, vì không có đối chứng bank khác với cùng loss/miner.

### 6.3. hybrid_fixed — attention schedule từ map ban đầu

**Cách hoạt động:** giữ negative loss và hard top-2; Qwen tạo map ở bước 0, xây schedule rồi giữ map đó trong toàn bộ 300 bước. Crop vẫn thay đổi theo các entry của schedule, nên “fixed” không có nghĩa dùng một crop cố định.

**So sánh bổ trợ:** hybrid_fixed − hybrid_negative.

**Kỳ vọng:** dùng vùng được attention định hướng có lợi hơn crop thông thường.

**Giới hạn:** so sánh này thêm đồng thời việc chọn vị trí và cơ chế lịch vùng; nó không tách riêng đóng góp của từng cơ chế. Đối chứng random-region mới trực tiếp kiểm tra vai trò vị trí attention trong lịch vùng.

### 6.4. hybrid_dynamic — H3: giá trị của refresh

**Cách hoạt động:** giống hybrid_fixed về loss và hard negatives, nhưng tái tính map từ ảnh adversarial hiện tại sau mỗi 30 bước. Caption guide vẫn là source caption factual cố định.

**So sánh chính:** hybrid_dynamic − hybrid_fixed.

**Kỳ vọng:** map cập nhật theo ảnh đang thay đổi giúp chọn vùng phù hợp hơn map ban đầu.

**Kết luận được phép:** chênh lệch hỗ trợ lợi ích của refresh trong schedule hiện tại. Cần báo cả thời gian vì dynamic có 10 lượt attention phục vụ tối ưu thay vì 1.

### 6.5. hybrid_random_region — H2: vị trí attention có ích không?

**Cách hoạt động:** giữ FOA, MPC, hard negatives và nhịp refresh. Code vẫn tạo schedule từ attention, nhưng thay vị trí box bằng vị trí ngẫu nhiên hợp lệ, giữ kích thước/phase tương ứng.

**So sánh chính:** hybrid_dynamic − hybrid_random_region.

**Kỳ vọng:** nếu dynamic tốt hơn, vị trí vùng do attention chọn có giá trị vượt việc chỉ áp dụng một lịch crop theo diện tích/hình dạng.

**Giới hạn:** kết quả không chứng minh attention là causal attribution. Random-region còn phụ thuộc hình dạng/lịch từ map, nên không được mô tả là hoàn toàn loại bỏ Qwen.

<!--
### 6.6. hybrid_random_neg — H1: hard mining có ích không?

**Cách hoạt động:** giữ dynamic region và cùng bank bốn caption. Mỗi kỳ mining chọn ngẫu nhiên hai caption bằng RNG riêng thay vì chọn theo thứ hạng similarity.

**So sánh chính:** hybrid_dynamic − hybrid_random_neg.

**Kỳ vọng:** chọn negatives đang dễ bị nhầm lẫn hơn giúp transfer tốt hơn chọn ngẫu nhiên.

**Điều kiện:** bank phải lớn hơn số negative được chọn để hai chính sách khác nhau có ý nghĩa. Policy hiện yêu cầu bank 4, chọn 2. Vẫn phải báo bank size thực tế; điều kiện plan yêu cầu báo toàn cohort và subset bank_size > 2, không âm thầm loại các mẫu bank nhỏ.

**Giới hạn:** đây là kiểm tra hard selection **trong bối cảnh dynamic region**, chưa chứng minh lợi ích giống hệt khi không dùng region. -->

### 6.7. h4_mpc_zero — H4: đóng góp của MPC term

**Cách hoạt động:** dùng variant hybrid_plain nhưng đặt trọng số MPC bằng 0. Không negative loss thực tế vì variant không có negatives. Objective hữu hiệu là FOA.

**So sánh chính:** primary_ablations/hybrid_plain − h4_mpc_zero/hybrid_plain.

**Kỳ vọng:** nếu cấu hình có MPC tốt hơn, MPC term bổ sung tín hiệu có ích cho FOA trong hybrid hiện tại.

**Giới hạn:** đây là đối chứng trong cùng bộ khung hybrid, không mặc nhiên thay thế một lần reproduce FOA reference đầy đủ. Code hiện vẫn forward/tính MPC rồi nhân 0; phép tính dư ảnh hưởng runtime dù không đóng góp objective khi mọi giá trị hữu hạn.

<!--
### 6.8. h4_image_only — H4: đóng góp text trong anchors

**Cách hoạt động:** giữ trọng số MPC bằng 1 nhưng đặt lambda = 1, do đó anchors MPC chỉ còn embedding ảnh. FOA, crop và các thiết lập khác giữ nguyên.

**So sánh chính:** primary_ablations/hybrid_plain − h4_image_only/hybrid_plain.

**Kỳ vọng:** image–text fusion lambda = 0,6 tốt hơn anchors chỉ ảnh lambda = 1. -->

**Giới hạn:** “image-only” chỉ nói về **anchors MPC của đối chứng này**. Không được diễn giải thành mọi phần của toàn pipeline/evaluation không dùng text. So sánh cũng chưa tìm được lambda tối ưu.

## 7. Phép so sánh và bằng chứng cần báo cáo

| Giả thuyết trong plan | Treatment            | Control                    | Chênh lệch cần báo                     |
| --------------------- | -------------------- | -------------------------- | -------------------------------------- |
| H1_negative_loss      | hybrid_negative      | hybrid_plain               | Delta ASR, Delta AvgSim                |
| H1_hard_mining        | hybrid_dynamic       | hybrid_random_neg          | Delta ASR, Delta AvgSim                |
| H2_attention_location | hybrid_dynamic       | hybrid_random_region       | Delta ASR, Delta AvgSim                |
| H3_refresh            | hybrid_dynamic       | hybrid_fixed               | Delta ASR, Delta AvgSim, Delta runtime |
| H4_mpc_term           | primary/hybrid_plain | h4_mpc_zero/hybrid_plain   | Delta ASR, Delta AvgSim                |
| H4_image_text_fusion  | primary/hybrid_plain | h4_image_only/hybrid_plain | Delta ASR, Delta AvgSim                |

Trong bảng, primary là viết tắt của primary_ablations. Mỗi phép so sánh thực hiện riêng trên từng victim, ghép theo cùng sample_id và hash ảnh.

Một kết quả dương nhất quán trên victim và có khoảng bất định phù hợp là bằng chứng ủng hộ giả thuyết trong phạm vi cohort này. Kết quả gần 0 hoặc âm không nên bị thay bằng các ví dụ thành công riêng lẻ. Một seed không đủ để khẳng định độ bền qua seed hoặc tính phổ quát.

## 8. Pipeline chạy trên Molab

Code orchestration và attack nằm trong notebook v4.3.3; evaluation worker thực thi protocol từ notebook v3.1 cùng thư mục trong interpreter riêng.

Thứ tự controller:

1. Kiểm tra dữ liệu, model locks, code identity và khả năng ghi/đọc lại Drive.
2. Smoke 2 cặp ảnh × 300 bước và evaluation smoke; dùng checkpoint smoke hợp lệ khi resume.
3. primary_ablations / hybrid_plain.
4. h4_mpc_zero / hybrid_plain.
5. h4_image_only / hybrid_plain.
6. primary_ablations / hybrid_negative.
7. primary_ablations / hybrid_fixed.
8. primary_ablations / hybrid_dynamic.
9. primary_ablations / hybrid_random_region.
10. primary_ablations / hybrid_random_neg.

**Sau mỗi cấu hình đủ 200 ảnh:** chạy caption LLaVA, caption InternVL, GPT judge LLaVA, GPT judge InternVL; sau đó mới chuyển sang cấu hình tiếp theo. Không đợi cả tám cấu hình xong mới đánh giá.

Nếu GPT gặp quota pause, controller có thể giữ scores để resume và tiếp tục công việc còn được phép; cấu hình chưa đủ điểm không được ghi thành đánh giá hoàn chỉnh.

Attack xử lý ảnh theo thứ tự để giữ FOA/RNG. Captioning dùng batch 64 trên một GPU và chia nhỏ khi OOM. GPT judge chạy API với tối đa 4 worker, governor 12 RPM/8.000 TPM theo plan; tăng VRAM không tăng trực tiếp tốc độ GPT API.

Trên giao diện Molab đã dọn, Cell 1 khởi động/resume tiến trình nền; Cell 2 đọc trạng thái/log và cập nhật khoảng 10 giây. Việc Cell 1 đã chạy xong không đồng nghĩa controller đã dừng.

### Checkpoint và resume

Root runtime: /marimo/molab_v433_campaign. Các artifact chính:

```text
molab_v433_campaign/
  controller_pid.json
  controller.log
  progress.json
  attack/
    campaign_state.json
    work/<job>/<run>/
      run_config.json
      pairs.json
      text_records.json
      <variant>/
        status.json
        manifest.jsonl
        errors.jsonl
        samples/<sample_id>/
          complete.json
          state.json
          attack_stats.json
          steps.jsonl
          checkpoints/
          maps/          # variant có region
          schedules/     # variant có region
  evaluation/
    evaluation_state.json
    runs/<configuration>__<victim>/<run>/
      caption_items.jsonl
      evaluation/
        scores.jsonl
        status.json
        summary.json
```

Ảnh output được ghi theo manifest của run; không suy đoán tên đường dẫn ảnh ngoài manifest. Một sample chỉ được công nhận hoàn tất sau khi ảnh và state đã lưu, có complete record và kiểm tra hash. Driver upload ZIP sample và metadata, đọc lại kiểm tra SHA-256.

Evaluation lưu caption/score để bỏ qua phần hợp lệ khi resume; backup định kỳ khoảng 120 giây và ở các ranh giới kết thúc/pause. Caption của ảnh có cùng hash và cùng protocol có thể được tái sử dụng.

Resume đòi hỏi đúng config/code/inputs/packages và model locks. Không chỉnh fingerprint để ép trộn code mới vào kết quả cũ. PNG ở bước 30, 60… là artifact chẩn đoán; implementation hiện không khôi phục một ảnh giữa chừng chỉ từ PNG đó. Sau timeout, ảnh chưa hoàn tất có thể phải chạy lại từ checkpoint ranh giới ảnh trước.

## 9. Evaluation và ý nghĩa của metric

### 9.1. Hai victim

- LLaVA-1.5-7B.
- InternVL3-8B.

Victim sinh caption cho ảnh target và ảnh adversarial bằng cùng prompt: một câu ngắn, không quá 20 từ; giới hạn sinh 64 token trong protocol. Model local dùng cấu hình quantization của evaluation v3.1, hiện là NF4. Target caption dùng để judge là caption do **victim tương ứng** sinh, không phải mặc định lấy caption factual của surrogate.

InternVL3-8B khác checkpoint InternVL3-1B dùng làm captioner trong attack nhưng cùng họ model, cần công bố điều này khi nói về transfer. Qwen3-VL-8B đã dùng sinh/audit negative và attention, vì vậy evaluation Qwen nếu bổ sung chỉ là surrogate diagnostic, không phải victim độc lập. Plan hiện không chạy diagnostic đó.

### 9.2. GPT-4o judge

GPT-4o nhận hai caption target/adversarial và trả similarity trong [0,1], temperature = 0, max output = 100 token. Nó không nhận hai ảnh để xác minh trực tiếp.

Quy tắc thành công của protocol hiện tại:

$$
\mathrm{success}_i = \mathbb{1}[q_i \ge 0{,}3].
$$

Với tập V các cặp được judge hợp lệ:

$$
\mathrm{ASR}_{valid}
=\frac{1}{|V|}\sum_{i\in V}\mathrm{success}_i,\qquad
\mathrm{AvgSim}_{valid}
=\frac{1}{|V|}\sum_{i\in V}q_i.
$$

Coverage = |V|/200. Chỉ khi coverage = 100% mới gọi ASR trên toàn cohort là kết quả hoàn chỉnh. Trong lúc chạy, success_count/200 có thể báo như lower bound, không được âm thầm tính lỗi API thành score 0 hoặc loại lỗi rồi công bố ASR như đủ 200 ảnh.

**Giới hạn quan trọng với H1:** rubric judge chú trọng chủ thể/ngữ nghĩa tổng quát và bỏ qua khác biệt chi tiết nhỏ. Ngưỡng 0,3 còn chấp nhận mức liên quan tương đối rộng. Vì vậy ASR cao theo protocol này chưa đủ chứng minh attack điều khiển đúng màu sắc, số lượng hoặc quan hệ nhân quả cụ thể. Để kết luận về fine-grained counterfactual discrimination cần thêm đánh giá thuộc tính/quan hệ hoặc kiểm tra thủ công; báo cáo này không tự thay đổi rubric đang chạy.

### 9.3. Phân tích thống kê và mẫu đối chứng

- Báo ASR, AvgSim, coverage, số lỗi và thời gian riêng cho từng cấu hình × victim.
- Ghép treatment/control trên cùng sample_id hợp lệ; báo kích thước tập ghép và phần thiếu.
- Báo số cặp chỉ treatment thành công, chỉ control thành công, cả hai và không bên nào thành công.
- Dùng chênh lệch theo cặp cho ASR/score; khoảng bất định cần xét việc source/target có thể lặp lại theo hash, không mặc định mọi cặp độc lập.
- Sáu so sánh là các phép kiểm định có liên quan; công bố cách xử lý nhiều phép so sánh nếu báo p-value.
- Nên có clean-source → target similarity/ASR để biết mức khớp sẵn có trước attack. Cohort có ảnh source/target trùng hoặc gần nghĩa cần được công bố theo dữ liệu, không loại sau khi nhìn kết quả.

## 10. Kiểm tra tính dư và hiệu năng

Đây là audit implementation, chưa phải các thay đổi đã được áp dụng.

| Điểm kiểm tra                      | Kết luận                                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| MPC trong h4_mpc_zero              | Vẫn được tính mỗi bước rồi nhân 0; có thể bỏ nhánh sau kiểm chứng tương đương và quản lý version      |
| Text encoders trong h4_image_only  | Vẫn encode source/target caption dù hệ số text bằng 0; không cần cho objective hiện tại của đối chứng |
| Scoring trong hybrid_random_neg    | 30 lượt/ảnh; không quyết định random IDs, nhưng đang dùng cho finite checks và trace                  |
| Attention bước 300                 | Chẩn đoán, không cập nhật delta; bỏ sẽ mất artifact attention cuối                                    |
| Attention trong random-region      | Vẫn cần cho lịch/kích thước box của đối chứng hiện tại                                                |
| FOA ảnh đầy đủ                     | Cần cho lịch sử trọng số; không được bỏ như một forward thừa                                          |
| Negative bank dùng qua các variant | Đã cache; sinh/audit lại khi policy/input không khớp là kiểm tra cần thiết                            |
| FOA/MPC dùng chung trọng số CLIP   | Giảm trùng weights; không tự loại các forward riêng                                                   |

Có thể nghiên cứu gộp vision forward trên cùng crop rồi trích riêng đặc trưng FOA/MPC. Hai loại vector không giống nhau nên không được thay trực tiếp. Có thể cache normalized text features và anchors cố định khi model/input/preprocessing khớp.

### Số đo thực tế đã thu ngày 17/10/2026

Đo ở lượt h4_mpc_zero/hybrid_plain trên RTX PRO 6000 Blackwell Server Edition:

| Chỉ số                                                 |             Giá trị đã đo |
| ------------------------------------------------------ | ------------------------: |
| VRAM                                                   | 8.5/95.6 GiB, khoảng 8,9% |
| GPU busy trung bình, 180 mẫu cách nhau khoảng 1 giây   |                     55,7% |
| GPU busy cao nhất trong cửa sổ đo                      |                       90% |
| Thời gian tối ưu trung bình của 10 ảnh gần nhất lúc đo |            77,45 giây/ảnh |
| Khoảng thời gian trung bình giữa hai ảnh hoàn tất      |               104,72 giây |
| Chênh lệch ngoài vòng tối ưu                           |     Khoảng 27,27 giây/ảnh |

Artifact đo ở runtime gồm gpu_usage_probe_long.csv và gpu_timing_breakdown.json. Đây là số đo cục bộ tại một thời điểm, không phải tổng kết cả tám cấu hình. Phần chênh lệch gồm xuất artifact, kiểm tra, lưu state/manifest và backup Drive; không quy toàn bộ 27,27 giây cho duy nhất network.

VRAM sử dụng thấp không đồng nghĩa GPU chỉ tính toán ở mức 8,9%. GPU busy cũng không chứng minh đã đạt đỉnh FLOPS hay occupancy. [NVIDIA mô tả các chỉ số nvidia-smi](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/kw/linux).

Ưu tiên tăng tốc là loại nhánh không đóng góp objective, benchmark sao lưu nền có hàng đợi giới hạn/checksum/flush, rồi benchmark các ablation độc lập chạy song song. Tăng batch attack ảnh thông thường sẽ đụng vào FOA history và RNG. Caption batch hiện là 64; evaluation v3.1 còn assert batch không quá 64, nên thử 128 đòi hỏi thay code và kiểm chứng riêng, không chỉ đổi một tham số ở UI.

Mọi tối ưu cần ghi version, so sánh objective/gradient/RNG/ảnh cuối và kiểm tra resume trước khi dùng cho cohort chính. Các số tốc độ của bản cũ không được gán cho bản tối ưu chưa benchmark.

## 11. Mẫu bảng kết quả để hoàn thiện sau chạy

Điền một bảng riêng cho mỗi victim. Dấu “—” nghĩa là chưa điền kết quả được xác minh, không phải score bằng 0.

| Cấu hình                       | Attack hoàn tất / 200 | Judge hợp lệ / 200 | ASR toàn cohort | AvgSim | Tổng thời gian |
| ------------------------------ | --------------------: | -----------------: | --------------: | -----: | -------------: |
| primary / hybrid_plain         |                     — |                  — |               — |      — |              — |
| h4_mpc_zero / hybrid_plain     |                     — |                  — |               — |      — |              — |
| h4_image_only / hybrid_plain   |                     — |                  — |               — |      — |              — |
| primary / hybrid_negative      |                     — |                  — |               — |      — |              — |
| primary / hybrid_fixed         |                     — |                  — |               — |      — |              — |
| primary / hybrid_dynamic       |                     — |                  — |               — |      — |              — |
| primary / hybrid_random_region |                     — |                  — |               — |      — |              — |
| primary / hybrid_random_neg    |                     — |                  — |               — |      — |              — |

Kèm bảng chênh lệch treatment–control của sáu so sánh, khoảng bất định, số cặp ghép và phân tích failure/coverage. Chỉ sau đó mới viết kết luận giả thuyết được ủng hộ, chưa được ủng hộ hay chưa đủ dữ liệu. Không suy ra “chứng minh H1” chỉ từ surrogate objective tăng hoặc một vài ảnh ví dụ.

## 12. Nguồn và truy vết

- [Notebook attack v4.3.3](kltn-attack-v4.3.3-h1-cfvlm.ipynb): MOLAB_PLAN, VARIANTS, AttackModels.anchors, attack_objective, run_one, prepare_text_records, record_attention, run_experiment và controller.
- [Notebook evaluation v3.1](text_gen_eval_v3.1.ipynb): CaptionGenerator, GPTScorer.compute_similarity, calc_metrics và điều kiện thành công.
- [CF-VLM paper, bản v1](https://arxiv.org/html/2506.17267v1): nguồn tham khảo chính sách can thiệp counterfactual; objective attack trong báo cáo là implementation của dự án.
- [CF-VLM official code](https://github.com/HCP-AI-Research-Lab/CF-VLM).
- [FOA author repository](https://github.com/jiaxiaojunQAQ/FOA-Attack) và [MPC author repository](https://github.com/LiYuanBoJNU/MPCAttack): notebook nhúng definitions và provenance được khóa; khi tái lập ưu tiên commit/hash trong run_config.
- Artifact thực nghiệm: run_config.json, model_metadata.json, text_records.json, manifest.jsonl, state.json, steps.jsonl và evaluation/scores.jsonl.
