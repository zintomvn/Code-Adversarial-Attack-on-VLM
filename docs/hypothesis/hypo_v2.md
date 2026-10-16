# Báo cáo giả thuyết nghiên cứu — v2

**Chủ đề:** XAI và counterfactual hard negatives cho targeted black-box transfer attack từ CLIP sang LVLM.  
**Ngày cập nhật:** 15/10/2026.  
**Implementation được mô tả:** [kltn-attack-v4.3-h1.ipynb](../../notebooks/attacks/kltn-attack-v4.3-h1.ipynb).  
**Tài liệu trước:** [hypo_v1.md](hypo_v1.md).

## 1. Tóm tắt và trạng thái nghiên cứu

Câu hỏi nghiên cứu là: **Khi giữ ensemble CLIP và perturbation budget cố định, việc bổ sung counterfactual hard negatives cùng chuỗi vùng lấy từ Qwen causal self-attention có cải thiện targeted transfer sang các LVLM held-out hay không?**

Notebook v4.3 tạo một khung thí nghiệm gồm FOA reference, MPC-CLIP, full MPC tùy chọn và các biến thể hybrid. Hybrid kết hợp feature alignment của FOA, contrastive objective của MPC và negative-concept loss. Qwen3-VL-8B đảm nhiệm hai công việc: sinh counterfactuals của target và lấy attention của source caption trên ảnh adversarial hiện tại để chọn vùng crop.

Trạng thái bằng chứng tại thời điểm viết report:

- Đã có implementation, kiểm tra parity trên tensor/encoder nhỏ, kiểm tra attention và kiểm tra artifacts/resume.
- Đã có CUDA smoke test với encoder nhỏ trên RTX 4060 Laptop 8 GB.
- Metadata của notebook ghi nhận chưa chạy full pretrained-model smoke trên Kaggle T4 ×2. Report này chưa được cung cấp và đối chiếu kết quả attack/evaluation thực tế từ Kaggle.
- Chưa có ASR, AvgSim hay phân tích thống kê đủ để kết luận hybrid tốt hơn FOA hoặc MPC.
- Các loss và thành phần mượn từ nghiên cứu trước không được xem là novelty độc lập. Tính mới của tổ hợp và cơ chế tương tác vẫn cần rà soát literature và kiểm chứng thực nghiệm.

Report phân biệt ba mức: **cơ chế đã triển khai**, **giả thuyết cần kiểm chứng**, và **thực nghiệm hoặc chẩn đoán đề xuất bổ sung**. Những mục ở mức thứ ba không có nghĩa notebook hiện tại đã tự động thực hiện chúng.

## 2. Những thay đổi so với v1

Phần mô tả pipeline trong v1 thuộc thiết kế cũ. Khi phân tích notebook v4.3, sử dụng các điều chỉnh dưới đây.

| Nội dung | Thiết kế trong v1 | Implementation v4.3 |
|---|---|---|
| XAI | Gradient-weighted attention rollout trong CLIP | Causal self-attention trong decoder Qwen3-VL-8B |
| Guide chọn vùng | Source/negative concept margins trong CLIP | Source caption cố định, teacher-force vào Qwen |
| Negative generator | InternVL trong pipeline cũ | Qwen nhận target image và target caption |
| Temporal mechanism | Fresh map, EMA và memory credit | Fresh map và stage sequencing; không có EMA/memory credit |
| FOA control | FOA-derived objective có thay đổi | Reference lấy thân extractor/loss từ snapshot upstream, kiểm tra update parity |
| Target của FOA | Cached anchors trong thiết kế cũ | Target crop được lấy lại ở từng bước |
| Full/crop | Có full/crop averaging | FOA full score cập nhật history trước, crop score quyết định gradient khi bật source crop |
| Pixel support | Có explicit support mask | Gradient đi qua crop; không thêm pixel mask |
| MPC | Chủ yếu objective trên từng CLIP | Concatenate representations rồi tính contrastive objective |
| Negative filtering | Có CLIP filtering trong v1 | Parser kiểm tra cấu trúc; không có bộ lọc semantic bằng CLIP như v1 |
| Ablation names | `h1_full`, `h1_no_hard`, ... | `hybrid_plain`, `hybrid_negative`, `hybrid_fixed`, `hybrid_dynamic`, ... |

Vì vậy, không dùng các diễn giải về CLIP rollout, negative-conditioned grounding hoặc memory trong v1 để giải thích kết quả v4.3.

## 3. Bài toán, threat model và vai trò model

### 3.1. Bài toán

Cho source image $x_s$ và target image $x_t$, tìm perturbation $\delta$ để victim mô tả ảnh adversarial gần nội dung target, đồng thời giữ nhiễu nhỏ.

Trong notebook, attack tensor dùng miền pixel $[0,255]$:

$$
\|\delta\|_\infty\le16,\qquad \alpha=1,\qquad T=300.
$$

Ở từng bước, ảnh đưa vào các adapter là $x_s+\delta$. Các normalizer của tác giả xử lý clamp/rescale bên trong. Ảnh trả ra cuối attack là:

$$
\widehat{x}_a=\operatorname{clip}\left(\frac{x_s+\delta}{255},0,1\right).
$$

Budget tương đương $16/255$ trong miền $[0,1]$. Không thay quy trình này bằng clamp ảnh ở mỗi bước nếu mục tiêu là giữ reference behavior.

### 3.2. Threat model

Trong vòng tối ưu, model weights được đóng băng và chỉ pixels được cập nhật. Gradient lấy từ surrogate CLIP; full MPC có thêm gradient từ nhánh ảnh InternVL và DINOv2. Qwen cung cấp attention và text proposals dưới `no_grad`, không tham gia backprop pixel loss.

Không sử dụng gradient, attention hay phản hồi của victim để chọn negative, chọn vùng hoặc tối ưu nhiễu. Qwen là **surrogate có truy cập nội bộ**, vì vậy kết quả trên đúng checkpoint Qwen dùng để lấy attention phải được báo riêng; không xem nó là held-out victim độc lập. Tương tự, cần ghi rõ model ID/revision khi victim cùng họ với InternVL captioner.

### 3.3. Model và representation

| Model/checkpoint | Vai trò mặc định | Tham gia pixel gradient? |
|---|---|---|
| `openai/clip-vit-base-patch16` | FOA hidden features; MPC projected image/text embeddings | Có, qua image branch |
| `openai/clip-vit-base-patch32` | Như trên | Có, qua image branch |
| `laion/CLIP-ViT-G-14-laion2B-s12B-b42K` | Như trên | Có, qua image branch |
| `OpenGVLab/InternVL3-1B-hf` | Sinh source/target captions | Không trong cấu hình MPC-CLIP/hybrid mặc định |
| `Qwen/Qwen3-VL-8B-Instruct` | Sinh counterfactuals và lấy attention map | Không |
| `OpenGVLab/InternVL3-1B-hf` — full MPC | Thêm multimodal image features | Có khi bật `mpc_full_reference` |
| `facebook/dinov2-base` — full MPC | Thêm self-supervised image features | Có khi bật `mpc_full_reference` |

**Text encoder của MPC là text tower CLIP tương ứng với từng image encoder.** InternVL là caption generator, không phải một text encoder độc lập có embedding được cộng trực tiếp vào mọi CLIP. Hai adapter FOA/MPC chia sẻ CLIP weights nhưng giữ preprocessing và representation cần cho từng phương pháp.

FOA dùng global/local hidden features trong từng backbone. MPC và negative loss dùng projected embeddings cùng không gian ảnh–text của từng CLIP. Không so sánh trực tiếp patch hidden features với text embeddings nếu chưa có projection phù hợp.

## 4. Nền tảng FOA và MPC trong notebook

### 4.1. FOA reference

Reference dựa trên entrypoint `generate_adversarial_samples_foa_attack.py`, ensemble B16/B32/Laion và K-means $K=10$. Nó không dùng entrypoint tự chọn số cluster trong `FOAttack.py`. Snapshot upstream được nhúng trong notebook để kiểm tra và chạy độc lập với checkout hiện tại. [FOA author repository](https://github.com/jiaxiaojunQAQ/FOA-Attack).

Với backbone $m$, gọi $g_m(x)$ là global feature và $P_m(x)$ là các local cluster centers. Score của một lần gọi loss có dạng:

$$
q_m=\operatorname{OTSim}(g_m(x_t'),g_m(x_a'))
+0.2\operatorname{OTSim}(P_m(x_t'),P_m(x_a')).
$$

OT dùng cosine similarity giữa vectors và transport plan tính dưới `no_grad`. Source K-means giữ đường truyền autograd của implementation được nhúng; không thay toàn bộ local branch bằng centers đã detach hoặc một custom clustering loss. Target features được tạo dưới `no_grad`.

Để thể hiện đúng trạng thái, dùng chỉ số **lần gọi loss** $u$, thay vì giả định mỗi iteration chỉ có một lần gọi:

$$
r_m^{(u)}=\frac{q_m^{(u)}}{q_m^{(u-1)}+10^{-8}},\qquad
w_m^{(u)}=M\frac{\exp(r_m^{(u)})}{\sum_j\exp(r_j^{(u)})},\qquad
S_{FOA}^{(u)}=\sum_m w_m^{(u)}q_m^{(u)}.
$$

Với $M=3$, tổng weights là 3. Lần gọi đầu tiên khởi tạo previous scores bằng current scores. History được cập nhật sau từng lần gọi và được giữ qua các mẫu trong cùng variant.

Khi source crop bật, mỗi iteration thực hiện:

1. Lấy target crop mới và cập nhật target ground truth.
2. Tính full-image FOA score để cập nhật history/weights.
3. Lấy source crop và tính crop score với history vừa cập nhật.
4. Dùng crop score để lấy gradient; không trung bình full score và crop score.
5. Gradient ascent, rồi clamp $\delta$ vào budget.

Notebook bỏ graph không cần dùng của full pass nhưng giữ thứ tự tính và thay đổi state. Kiểm tra với upstream trên encoder nhỏ cho thấy output và history khớp trong các trường hợp đã thử. Điều này chưa chứng minh bitwise parity của tất cả pretrained encoders trên mọi GPU/runtime.

### 4.2. MPC-CLIP và full MPC

Trong implementation MPC, source/target captions được InternVL sinh rồi mã hóa bằng text towers CLIP. Với từng backbone:

$$
j_{s,m}=\lambda z_{s,m}+(1-\lambda)t_{s,m},\qquad
j_{t,m}=\lambda z_{t,m}+(1-\lambda)t_{t,m},\qquad \lambda=0.6.
$$

MPC-CLIP concatenate các block:

$$
Z_s=[j_{s,1};j_{s,2};j_{s,3}],\quad
Z_t=[j_{t,1};j_{t,2};j_{t,3}],\quad
Z_a=[z_{a,1};z_{a,2};z_{a,3}].
$$

Full MPC bổ sung InternVL image features sau mean pooling và DINOv2 image features theo đúng thứ tự của code tác giả. Không lấy trung bình ba CE độc lập để thay cho CE của concatenated representation.

Các fused CLIP blocks không được normalize lại riêng trước concatenate trong implementation đang tái hiện. Toàn vector được normalize trong contrastive score. Đây là lựa chọn theo code, cần phân biệt với cách trình bày normalization từng block trong Eq.5 của paper. [MPC paper §3.2](https://arxiv.org/html/2603.04846v1#S3.SS2), [MPC author code](https://github.com/LiYuanBoJNU/MPCAttack/blob/7b195a5bcbaeaf1e8534d101199810a2782caad3/generate_adversarial_examples_MPCAttack.py).

Objective được maximize là negative cross-entropy:

$$
S_{MPC}=-\operatorname{CE}\left(
\left[\frac{\omega\cos(Z_a,Z_t)}{\tau},
\frac{\cos(Z_a,Z_s)}{\tau}\right],0\right),
\qquad \omega=2,\quad\tau=0.2.
$$

Source representation là negative của MPC. Các counterfactual target captions ở phần sau là một ngân hàng negative khác, được đưa vào loss riêng.

MPC-only giữ momentum 0.9 và initialization `16 / 255 * randn_like(source_cpu)` trước khi chuyển delta lên GPU. Scale initialization này nhỏ so với miền attack $[0,255]$, nhưng được giữ như author code; không diễn giải nó thành Gaussian có standard deviation 16 pixel.

`mpc_clip` là ablation thiếu hai nhánh image features InternVL/DINO, nên kết quả của nó không đại diện cho kết quả full MPC trong paper. `STEPS != 300` được ghi là short-run diagnostic.

## 5. H1 — Counterfactual hard negatives giúp phân biệt target semantics

### 5.1. Phát biểu và cơ chế

**H1:** Ngoài target alignment, ép ảnh adversarial phân biệt target caption với những caption chỉ khác một subject/object phrase có thể cải thiện targeted transfer.

Ví dụ minh họa, không phải output thí nghiệm:

```text
Target:     A cat is sleeping on a sofa.
Negative 1: A dog is sleeping on a sofa.
Negative 2: A rabbit is sleeping on a sofa.
```

Negative được định nghĩa **sai so với target mong muốn**. Nó không phải target mới để kéo ảnh tới, cũng không phải danh sách source concepts cần xóa.

Qwen nhận target image và target caption, trả JSON với `span`, `replacement`, `caption`. Parser kiểm tra span xuất hiện đúng một lần, caption đúng bằng phép thay span đó, nội dung không rỗng, không trùng positive/candidate đã nhận và không chứa control markers bị cấm.

Bank được cache một lần cho mỗi mẫu và dùng chung giữa variants. Giới hạn trên là 12 candidates; tối thiểu cần đủ `K_NEG`, mặc định 2. Generation có tối đa 3 attempts và không có fallback bằng template. Nếu thiếu negatives, variant cần negatives báo lỗi; runner không âm thầm bỏ negative loss.

Parser không chứng nhận replacement đúng là noun phrase, counterfactual sai về mặt hình ảnh hoặc không phải synonym. Notebook hiện không có CLIP semantic filtering như thiết kế v1. Do đó cần audit `text_records.json` và lưu phiên bản bank được dùng để phân tích.

### 5.2. Hard selection

Với candidate $i$ và projected embeddings đã normalize trong từng CLIP:

$$
h_i^{(t)}=\frac1M\sum_{m=1}^{M}\cos\left(z_{a,m}^{(t)},n_{i,m}\right),\qquad
\mathcal{K}_t=\operatorname{TopK}_i h_i^{(t)}.
$$

Chọn trên full adversarial image, không dùng victim feedback. Cập nhật selection mỗi 10 bước: trước update 1, 11, 21, ...; bank không được sinh lại trong vòng attack. Không backprop qua generation, scoring selection hoặc top-k.

**Điều kiện quan trọng:** nếu bank chỉ có đúng hai candidates và `K_NEG=2`, hard selection và random selection lấy cùng tập candidates. Khi đó thí nghiệm không kiểm chứng lợi ích hard mining, dù negative loss vẫn có tác dụng. Để kiểm chứng phần hard selection, cần bank lớn hơn `k_neg`, đồng thời báo phân bố kích thước bank thực tế. `MAX_NEGATIVES=12` là giới hạn trên, không bảo đảm Qwen sinh đủ 12 negatives hợp lệ.

### 5.3. Bốn negative losses đã triển khai

Đặt $s_m^+=\cos(z_{a,m},t_{t,m})$ và $s_{m,i}^-=\cos(z_{a,m},n_{i,m})$. Loss tính riêng trong không gian của từng backbone, sau đó trung bình trên ba CLIP. Các công thức dưới đây đều theo quy ước **minimize**:

| Cấu hình | Loss trên một backbone | Câu hỏi cần kiểm chứng |
|---|---|---|
| `infonce` | $\log(1+\sum_{i\in\mathcal{K}_t}\exp((s_{m,i}^- -s_m^+)/\tau_n))$ | So sánh tương đối target với toàn bộ selected negatives có hữu ích không? |
| `margin` | $\frac1k\sum_{i\in\mathcal{K}_t}[\mu+s_{m,i}^- -s_m^+]_+$ | Chỉ duy trì một khoảng cách phân biệt đủ lớn có tốt hơn CE không? |
| `hard_margin` | $[\mu+\max_{i\in\mathcal{K}_t}s_{m,i}^- -s_m^+]_+$ | Tập trung negative khó nhất có hiệu quả hơn trung bình nhiều negatives không? |
| `repulsion` | $\frac1k\sum_{i\in\mathcal{K}_t}s_{m,i}^-$ | Tác dụng đến từ negative repulsion hay cần positive–negative discrimination? |

Mặc định $\tau_n=0.2$, $\mu=0.2$. InfoNCE dùng `logsumexp` với một zero logit để tính ổn định. `repulsion` không có target-attraction term nội tại, vì vậy cần giữ FOA/MPC alignment đi kèm.

Các loss có scale khác nhau. So sánh cùng hệ số 0.5 là một cấu hình khởi đầu, không bảo đảm chúng đóng góp gradient tương đương. Nếu tune weights, chọn trên validation split và khóa lại trước khi đánh giá test.

### 5.4. Điều kiện hỗ trợ hoặc không hỗ trợ H1

So sánh `hybrid_plain` với `hybrid_negative` để đo negative-loss mechanism khi chưa dùng attention regions. So sánh `hybrid_dynamic` với `hybrid_random_neg` để đo hard selection trong cùng cơ chế dynamic attention.

H1 được hỗ trợ nếu cải thiện paired ASR/AvgSim trên held-out victims, nhất quán qua seeds và sau khi kiểm soát negative quality. Giảm negative loss hoặc tăng target similarity trong CLIP chưa đủ. Nếu hard selection không hơn random với bank đủ lớn, chưa có bằng chứng cho phần hard-mining của giả thuyết.

## 6. H2 — Vùng từ Qwen attention hiệu quả hơn vùng ngẫu nhiên

### 6.1. Phát biểu

**H2:** Với objective, lịch update và hình dạng crop tương đương, chọn vị trí vùng bằng source-caption-conditioned attention của Qwen có thể cải thiện transfer hơn chọn vị trí ngẫu nhiên.

Đây là giả thuyết về giá trị của attention như một **tín hiệu chọn vùng**, không phải tuyên bố attention tự nó chứng minh nguyên nhân của dự đoán.

### 6.2. Trích causal self-attention

Notebook tái sử dụng protocol trong [kltn-attack-v3.2.ipynb](../../notebooks/attacks/kltn-attack-v3.2.ipynb):

1. Chuyển ảnh adversarial hiện tại thành ảnh hợp lệ và chuẩn bị input Qwen ở độ phân giải 448.
2. Giữ guide là source caption đã cache; teacher-force caption này vào assistant suffix.
3. Chọn content-token query rows tại layer 29, đánh số từ 0.
4. Tính Q/K normalization, RoPE, grouped-query attention và causal mask theo decoder.
5. Softmax trên toàn bộ causal keys trước khi lấy image-token columns; không softmax chỉ trên image keys ngay từ đầu.
6. Trung bình heads, normalize vision attention theo từng selected token, rồi trung bình tokens để có probability map.
7. Tạo min–max map dùng cho hotspot ranking; lưu cả probability map và raw vision-attention mass để audit.

Image-token positions phải khớp merged spatial grid và loại marker tokens. Query rows là vị trí input của teacher-forced concept tokens; không đánh đồng với gradient attribution cho next-token prediction.

Guide text không thay đổi theo caption mới của Qwen trong từng iteration. Dynamic ở đây là **ảnh và attention map thay đổi**, không phải sinh lại concept dẫn vùng. Selected negatives chỉ đi vào loss, không conditioning attention map.

Qwen chạy NF4 với compute FP16 trong cấu hình dự kiến cho T4. Selected-row capture và early exit giảm chi phí lấy map; không có gradient của Qwen truyền vào pixels. Việc normalize một map có vision mass rất nhỏ vẫn có thể tạo phân bố nhìn rõ, nên cần xem raw mass trước khi diễn giải heatmap.

### 6.3. Chọn vùng và control

Hotspot search tìm các rectangular windows có tổng attention cao và diện tích gần mức yêu cầu. Có aspect-ratio constraint, area tolerance và IoU filtering. Khi thiếu box thỏa IoU, code lấy thêm các candidate tốt tiếp theo; do đó IoU cap không phải invariant tuyệt đối của mọi schedule.

Trong mỗi hotspot, thực hiện RandomResizedCrop với scale `(0.5, 0.9)` tương đối với hotspot, rồi resize về input resolution của CLIP. Mặc định `HOTSPOT_PROB=1.0`. Gradient chỉ đến các pixels liên quan đến crop qua phép biến đổi ảnh; không có support mask bổ sung.

`hybrid_random_region` giữ kích thước box và thứ tự area/stage, nhưng randomize vị trí. Variant này cũng refresh map như dynamic variant để tạo hình dạng schedule tương ứng. Tuy nhiên, randomization có thể làm thay đổi overlap và cumulative coverage. Vì vậy cùng area/shape từng crop chưa có nghĩa cùng union support qua toàn bộ attack.

Đối chiếu chính cho H2 là `hybrid_dynamic` với `hybrid_random_region`. Ngoài ASR/AvgSim, nên bổ sung phân tích cumulative crop coverage và overlap; notebook hiện lưu crop boxes để tính hậu kiểm, chưa tự xuất toàn bộ các chỉ số này.

## 7. H3 — Refresh attention tốt hơn giữ map cố định

### 7.1. Phát biểu

**H3:** Khi perturbation làm thay đổi evidence mà surrogate LVLM chú ý, cập nhật attention trên ảnh hiện tại có thể chọn vùng tiếp theo hiệu quả hơn schedule dựng từ ảnh source ban đầu.

SAGA cung cấp tiền lệ dùng một fixed pre-attack attention map để xây stage-wise region sequence. Fresh-map mechanism của notebook là một mở rộng cần kiểm chứng, không phải reproduction nguyên bản SAGA. [SAGA method](https://arxiv.org/html/2602.04356v2#S4).

### 7.2. Schedule thực tế

Mỗi lần dựng schedule, code xét 10 mức diện tích $a_n=n/10$, tìm tối đa 3 hotspots mỗi mức và nối chúng theo area group/sub-phase. `HOTSPOT_TOP_K` khác `K_NEG`.

Số entries $L$ là số box thực sự tìm được, không bị ép thành 30. Với grid đủ lớn, mức full area thường chỉ có một box nên schedule có thể có 28 entries. Trường hợp cụ thể cần đọc schedule đã lưu.

Index tại bước $t$, đánh số từ 0, là:

$$
b=\left\lfloor\frac{T}{L}\right\rfloor,\qquad
\operatorname{index}(t)=\min\left(\left\lfloor\frac{t}{b}\right\rfloor,L-1\right).
$$

Entry cuối nhận phần dư. Khi refresh, index vẫn tính từ bước tuyệt đối; không khởi động lại từ hotspot đầu tiên.

Với `STEPS=300`, `REFRESH_EVERY=30`:

- Map tại step 0 dùng từ update 1.
- Map tại step 30, 60, ..., 270 dùng từ update 31, 61, ..., 271.
- Map tại step 300 là diagnostic cuối, không được đưa lại vào optimization.
- `fixed` dùng map step 0 trong toàn bộ vòng tối ưu; vẫn có map cuối để quan sát.
- Không có EMA, visits/credit memory hoặc score-based discount vùng như v1.

### 7.3. Kiểm chứng và giới hạn

Đối chiếu `hybrid_fixed` với `hybrid_dynamic`. Giữ source guide, bank, objective, refresh negative selection và số gradient updates giống cấu hình; ghi nhận rằng ảnh tiến triển khác nhau có thể làm selected negatives khác nhau giữa methods.

Dynamic thường có 10 map computations phục vụ optimization cộng một map cuối; fixed có map đầu và map cuối. Cùng 300 updates không đồng nghĩa cùng wall-clock cost. Báo cả hiệu quả ở **matched update budget** và chi phí thời gian. Thí nghiệm matched wall-clock là phần đánh giá bổ sung, không phải khả năng benchmark tự động hiện có.

Nếu fresh maps không cải thiện held-out metrics hoặc chỉ tăng surrogate score với chi phí lớn, chưa đủ cơ sở giữ dynamic mechanism làm đóng góp chính. Attention có thể dịch chuyển theo perturbation nhưng vẫn theo dõi tín hiệu không transfer; cần phân tích những trường hợp thất bại này.

## 8. H4 — Joint alignment FOA/MPC bổ sung thông tin target semantics

**H4:** Kết hợp local/global feature alignment của FOA với source–target contrastive objective và text-fused anchors của MPC có thể bổ sung tín hiệu cho quá trình tối ưu.

Đây là giả thuyết về hiệu quả của hybrid trong setting đang nghiên cứu. Image/text fusion và contrastive target attraction/source repulsion đã là thành phần của MPC; không claim chúng là ý tưởng mới của report. [MPC methodology](https://arxiv.org/html/2603.04846v1#S3.SS2).

Hybrid maximize:

$$
J=w_F S_{FOA}+w_M S_{MPC}-w_N L_{neg},\qquad
(w_F,w_M,w_N)=(1,1,0.5).
$$

Dynamic ensemble weights chỉ dựa trên FOA scores, không dựa trên tổng hybrid objective. Negative loss được trung bình trên backbones; MPC loss tính trên concatenated representation. Vì scale khác nhau, weights bằng nhau không đồng nghĩa ảnh hưởng gradient bằng nhau.

Update có dạng:

$$
v_{t+1}=\beta v_t+\nabla_\delta J_t,\qquad
\delta_{t+1}=\operatorname{clip}_{[-\epsilon,\epsilon]}
\left(\delta_t+\alpha\operatorname{sign}(v_{t+1})\right).
$$

Hybrid mặc định $\beta=0$, initialization bằng zero như FOA. MPC-only dùng $\beta=0.9$ và Gaussian initialization như code MPC. Do đó so sánh trực tiếp FOA với MPC không tách riêng tác dụng fusion.

Hai kiểm chứng H4 có thể thực hiện bằng cấu hình hiện có, không cần thêm preset:

- Chạy `hybrid_plain` với `LOSS_WEIGHTS['mpc']=0` và với giá trị 1, giữ phần còn lại cố định, để đo việc thêm MPC term.
- Chạy `hybrid_plain` với `FUSION_LAMBDA=1.0` và `0.6` để so sánh image-only anchors với image/text anchors khi không có negative loss hay attention regions.

Mỗi cấu hình tạo run riêng, không dùng resume để đổi objective. Chưa có kết quả của các đối chiếu này trong report. Có thể bổ sung gradient-norm diagnostics từng term để giải thích tương tác, nhưng notebook hiện chỉ lưu score/loss components, không tự log gradient norm của từng term.

## 9. Ma trận ablation và kiểm soát công bằng

### 9.1. Các presets đã có

| Variant | FOA | MPC term | Counterfactual loss | Region | Vai trò |
|---|---|---|---|---|---|
| `foa_reference` | Có | Không | Không | Standard source crop | Baseline FOA |
| `mpc_clip` | Không | Có, CLIP-only | Không | Standard source crop | MPC ablation cùng ba CLIP |
| `mpc_full_reference` | Không | Có, đủ CLIP/InternVL/DINO | Không | Standard source crop | Full MPC reference tùy chọn |
| `hybrid_plain` | Có | Có, CLIP-only | Không | Standard source crop | Joint alignment control |
| `hybrid_negative` | Có | Có, CLIP-only | Hard top-k | Standard source crop | Đo negative-loss mechanism |
| `hybrid_fixed` | Có | Có, CLIP-only | Hard top-k | Fixed Qwen map | Attention sequencing control |
| `hybrid_dynamic` | Có | Có, CLIP-only | Hard top-k | Fresh Qwen maps | Cấu hình đề xuất mặc định |
| `hybrid_random_region` | Có | Có, CLIP-only | Hard top-k | Random vị trí, fresh-map shape schedule | Control vị trí vùng |
| `hybrid_random_neg` | Có | Có, CLIP-only | Random k từ cùng bank | Fresh Qwen maps | Control hard selection |

| Câu hỏi | Đối chiếu chính | Điều kiện cần |
|---|---|---|
| H1: negative loss | `hybrid_plain` ↔ `hybrid_negative` | Cùng objective nền/precision |
| H1: hard selection | `hybrid_dynamic` ↔ `hybrid_random_neg` | Bank thực tế lớn hơn k |
| H2: chọn vị trí | `hybrid_dynamic` ↔ `hybrid_random_region` | Báo area/shape và cumulative coverage |
| H3: refresh map | `hybrid_fixed` ↔ `hybrid_dynamic` | Báo thêm chi phí Qwen |
| H4: thêm MPC objective | `hybrid_plain`, weight MPC 0 ↔ 1 | Các thành phần khác cố định |
| H4: fusion | `hybrid_plain`, lambda 1.0 ↔ 0.6 | Không thêm negative/region trong đối chiếu này |
| Hiệu quả tổng thể | `hybrid_dynamic` ↔ FOA/MPC-CLIP/full MPC | Nêu rõ khác biệt surrogate access và compute |

`hybrid_negative` so với `hybrid_fixed` đo tác dụng của toàn cơ chế attention sequencing so với standard crop. Muốn lập luận riêng về **vị trí attention**, đối chiếu với random-region control phù hợp hơn.

### 9.2. Các yếu tố phải khóa hoặc báo rõ

- Cùng source/target pairs, thứ tự dữ liệu, captions, ngân hàng negatives, image preprocessing và perturbation budget.
- Cùng protocol victim generation, evaluator, prompt, success threshold và cách xử lý lỗi.
- Cùng seed khi so sánh paired variants. Lặp lại nhiều seeds cho kết luận tổng quát; reference seed khởi đầu là 2023.
- Giữ các RNG mining/region độc lập với crop/K-means. History FOA có trạng thái qua mẫu nên thay thứ tự mẫu cũng có thể thay kết quả.
- **Precision:** references giữ upstream CLIP FP32; hybrid mặc định `HYBRID_AMP=True`. Khi cần quy kết khác biệt cho thuật toán, thêm so sánh `HYBRID_AMP=False`. Nếu không đủ bộ nhớ, báo rõ precision confound thay vì gọi là so sánh chỉ khác objective.
- Full MPC có thêm image encoders; hybrid có thêm Qwen attention và negative-generation access. Không diễn giải mọi chênh lệch ASR thành tác dụng của một loss duy nhất.
- Ghi nhận actual bank size, negative-quality audit, candidate IDs/scores và tỷ lệ mẫu lỗi. Không chỉ giữ các mẫu có bank đẹp hoặc attack thành công.

`attack_stats.elapsed_seconds` hiện đo vòng attack và final attention diagnostic, không bao gồm toàn bộ model download/loading, text preparation và anchor construction trước timer. Khi báo end-to-end cost phải cộng hoặc đo riêng các giai đoạn này. Chi phí sinh bank được cache có thể amortize, nhưng cần công bố cách phân bổ chi phí.

## 10. Kế hoạch thực nghiệm và chỉ số

### 10.1. Trình tự đề xuất

1. **Kiểm tra chạy thật:** một cặp ảnh, 300 steps, ba variants mặc định. Xác nhận model loads, sample hoàn thành, budget và manifests trước khi mở rộng.
2. **Kiểm tra dữ liệu/semantics:** một tập nhỏ để audit source/target captions, counterfactual bank và attention maps. Chốt tiêu chí audit trước khi xem ASR.
3. **So sánh chính:** chạy cùng tập cặp ảnh cho ablations ở mục 9; full MPC chạy riêng khi có tài nguyên phù hợp.
4. **Chọn hyperparameters:** dùng validation pairs tách khỏi test; khảo sát từng yếu tố thay vì quét mọi tổ hợp ngay từ đầu.
5. **Đánh giá held-out:** khóa cấu hình, chạy nhiều seeds và nhiều victim checkpoints; báo paired differences và khoảng tin cậy.

Các sweep nhỏ có thể bắt đầu từ `K_NEG ∈ {1,2,4}`, negative weight `∈ {0,0.25,0.5,1}`, refresh interval `∈ {15,30,60}` và bốn negative losses. Đây là đề xuất thí nghiệm, không phải kết quả hoặc cấu hình đã được chứng minh tối ưu. Chỉ dùng k khi bank thực tế đủ lớn.

Nếu source/target images được tái sử dụng ở nhiều pairs, cần xét grouping khi chia validation/test và tính uncertainty; không xem mọi pair là quan sát độc lập một cách mặc định.

### 10.2. ASR và semantic similarity

Evaluator local [text_gen_eval_v2.ipynb](../../notebooks/text_gen_eval_v2.ipynb) đang đặt `JUDGE_MODEL='gpt-4o'`, `SUCCESS_THRESHOLD=0.3`, `SUCCESS_OPERATOR='>='`. [text_gen_eval_v3.ipynb](../../notebooks/text_gen_eval_v3.ipynb) cũng có các giá trị mặc định này tại thời điểm đọc. Đây là **protocol local**, không mặc định đồng nhất với mọi threshold trong paper.

Với tập pairs được judge hợp lệ $\mathcal{V}$ và similarity score $s_i$:

$$
ASR_{valid}=\frac{1}{|\mathcal{V}|}\sum_{i\in\mathcal{V}}\mathbb{1}[s_i\ge0.3],\qquad
AvgSim_{valid}=\frac{1}{|\mathcal{V}|}\sum_{i\in\mathcal{V}}s_i.
$$

Không chọn lại threshold sau khi xem kết quả test. Nếu muốn so sánh paper, dùng thêm đúng protocol paper và ghi nó thành một evaluation setting riêng.

Các evaluator đang tính similarity trên valid evaluated pairs. Nếu variants có số mẫu generation/evaluation lỗi khác nhau, ASR có mẫu số khác nhau và có thể bị selection bias. Báo tối thiểu:

- Tổng N pairs dự kiến, số attack artifacts hoàn thành và số pairs judge hợp lệ.
- ASR/AvgSim trên valid pairs, kèm coverage.
- Paired comparison trên tập hợp hợp lệ chung, kèm phân tích những mẫu bị loại.
- Có thể bổ sung operational success rate trên toàn bộ N, coi failure là không thành công; phải ghi rõ đây là chỉ số bổ sung bảo thủ, không đánh đồng lỗi API với semantic attack failure.

Đối với significance, đề xuất paired bootstrap trên cùng pair IDs hoặc phương pháp paired thích hợp, xét grouping khi có ảnh lặp. Báo effect size/CI theo từng victim; không chỉ báo trung bình pooled che mất victim có kết quả xấu.

### 10.3. Chẩn đoán bổ sung

Các artifacts hiện có hỗ trợ tính hậu kiểm: target/source similarity trajectory, negative selection changes, attention redistribution, crop coverage, runtime và saved perturbation budget. Các thống kê tổng hợp này cần script phân tích riêng nếu chưa có trong trace.

Nên chọn trước một tập examples để xem ảnh/heatmaps, gồm cả thành công và thất bại. Tăng attention contrast, giảm source similarity hay surrogate loss chỉ giải thích quá trình tối ưu; không thay thế metric trên victim.

## 11. Kết quả kiểm tra đã có và giới hạn bằng chứng

Các trạng thái dưới đây lấy từ `local_validation` metadata của notebook, không phải một lần rerun full attack trong lúc viết report.

| Hạng mục | Trạng thái ghi nhận | Phạm vi bằng chứng |
|---|---|---|
| Notebook validation và compile code cells | Đã kiểm tra khi tạo notebook | Cấu trúc notebook và cú pháp |
| Embedded upstream source hashes | Passed | Snapshot nhúng khớp hashes đã ghi |
| FOA loss/gradient/update/history/resume | Passed | Controlled tensors/encoder nhỏ; nhiều steps, crop on/off và hai mẫu |
| MPC anchors/score/gradient/init/momentum | Passed | Tensor kiểm soát đối chiếu author code |
| Negative losses, selection, region schedule, RNG | Passed | Kiểm tra số học và các invariants đã thiết kế |
| Selected Qwen rows so với eager attention | Passed | Qwen configuration nhỏ, random weights; kiểm tra causal masking |
| CUDA smoke | Passed | Tiny encoders trên RTX 4060 Laptop GPU |
| Artifacts/resume integration | Passed | Ba variants, hai synthetic pairs; budget PNG, manifests, ZIP và resume khớp hash |
| Full pretrained-model smoke trên T4 ×2 | Chưa được xác nhận trong metadata | Chưa chứng minh runtime/memory của toàn bộ cấu hình |
| ASR/AvgSim trên held-out victims | Chưa có số liệu được đối chiếu trong report | Chưa đánh giá hiệu quả nghiên cứu |

Runtime local của các kiểm tra được ghi là Torch `2.12.0+cu126` và Transformers `4.57.1`. Đây không phải xác nhận môi trường Kaggle có cùng phiên bản Torch. Full-model reproducibility cần lưu package versions/model revisions của run thực tế.

### Bảng kết quả thực nghiệm chờ điền

| Variant | Victim/revision | N dự kiến | N hoàn thành | N judge hợp lệ | ASR valid | AvgSim valid | Runtime/ảnh |
|---|---|---|---|---|---|---|---|
| `foa_reference` | Chưa đo | — | — | — | — | — | — |
| `mpc_clip` | Chưa đo | — | — | — | — | — | — |
| `mpc_full_reference` | Chưa đo | — | — | — | — | — | — |
| `hybrid_plain` | Chưa đo | — | — | — | — | — | — |
| `hybrid_negative` | Chưa đo | — | — | — | — | — | — |
| `hybrid_fixed` | Chưa đo | — | — | — | — | — | — |
| `hybrid_dynamic` | Chưa đo | — | — | — | — | — | — |
| `hybrid_random_region` | Chưa đo | — | — | — | — | — | — |
| `hybrid_random_neg` | Chưa đo | — | — | — | — | — | — |

Dấu “—” là chưa có số liệu, không có nghĩa metric bằng 0.

## 12. Vận hành, artifacts và tái lập

### 12.1. Cấu hình chạy thật trên Kaggle

Mục tiêu runtime là T4 ×2, ensemble attack ở GPU 0, Qwen ở GPU 1. Các flags tối thiểu từ kernel mới:

```python
RUN_ATTACK = True
SETUP_ONLY = False
INSTALL_DEPENDENCIES = True

RUN_SELF_CHECKS = True
GPU_SMOKE = False
NUM_SAMPLES = 1
STEPS = 300
RUN_VARIANTS = ['foa_reference', 'mpc_clip', 'hybrid_dynamic']

ATTACK_DEVICE = 'cuda:0'
CAPTION_DEVICE = 'cuda:0'
ATTENTION_DEVICE = 'cuda:1'
SAVED_TENSORS_ON_CPU = True
OFFLOAD_CLIP_DURING_QWEN = True
```

Cần sửa dataset paths, bật Internet và GPU trong Kaggle trước khi chạy. Nếu đã cài pinned packages, restart kernel rồi đặt `INSTALL_DEPENDENCIES=False`. Khi chỉ cài môi trường, dùng `RUN_ATTACK=False`, `SETUP_ONLY=True`; khi attack thật, phải trả `SETUP_ONLY=False`.

`RUN_ATTACK=False` là preflight. `GPU_SMOKE=True` chỉ chạy tiny encoders, không phải attack trên pretrained ensemble. `NUM_SAMPLES=1` cùng `RUN_ATTACK=True` vẫn là attack thật trên một pair.

Default hybrid AMP và CPU saved-tensor offload nhằm giảm bộ nhớ; chưa bảo đảm toàn bộ pretrained run sẽ vừa T4 ×2 trong mọi môi trường. Nếu thất bại, giữ error logs để phân biệt dependency, model load, text-generation, numerical và OOM failures; không đổi checkpoint hoặc bỏ backbone rồi gắn kết quả đó với cùng method label.

### 12.2. Dữ liệu và pairing

Nếu có `PAIR_MANIFEST`, dùng pairing và integer index từ manifest. Nếu không, dùng sorted ImageFolder pairing của hai thư mục `bigscale` và `target_images`. Hai thư mục cần cùng số mẫu; code không âm thầm truncate khi độ dài khác nhau.

Ảnh được RGB-convert, resize bicubic và center crop về 224 trước attack. Source/target pairing và hashes được lưu chung cho các variants. Thứ tự mẫu là một phần của điều kiện tái lập do FOA history không reset ở ranh giới mẫu.

### 12.3. Các artifacts cần giữ

| Artifact | Dùng để kiểm tra |
|---|---|
| `run_config.json`, `pairs.json` | Config, model revisions, source provenance, pair identity và fingerprint |
| `text_records.json` | Source/target captions, negative bank, generation attempts và lỗi |
| `model_metadata.json`, `self_checks.json` | Model/precision/backend metadata và kiểm tra kỹ thuật |
| `<variant>/manifest.jsonl` | Chỉ các mẫu hoàn thành, tương thích evaluation |
| `samples/<id>/source.png`, `target.png`, `adversarial.png` | Ảnh dùng để đánh giá; PNG lossless |
| `samples/<id>/steps.jsonl` | Objective components, FOA weights, negatives, crops và timing |
| `maps/`, `schedules/`, `checkpoints/` trong sample | Attention maps, vùng, checkpoint hashes và provenance |
| `samples/<id>/state.json`, `complete.json` | RNG/FOA history, completion marker và hashes |
| `errors.jsonl`, `status.json`, `summary.json` | Coverage, failures và trạng thái hoàn thành |

Đường dẫn ảnh trong manifest là tương đối với thư mục variant. Evaluation cần trỏ tới **một variant**, không trỏ root chứa nhiều manifests.

Resume chỉ ở ranh giới mẫu; mẫu chưa hoàn thành chạy lại từ đầu. Completed samples phải tạo thành prefix theo thứ tự pair. Runner khôi phục RNG/history và kiểm tra config/code/inputs/packages; đổi loss, weights, bank overrides hoặc thứ tự dữ liệu cần attack run mới. Thay evaluation threshold cần evaluation setting/run mới trên artifacts đã có, không tự động đòi hỏi sinh lại perturbation.

Dù tên flag là `CONTINUE_ON_SAMPLE_ERROR`, catch block hiện dừng vòng sample của variant khi có exception và có thể chuyển sang variant kế tiếp. Không diễn giải flag này thành bảo đảm bỏ qua mọi sample lỗi rồi tiếp tục cùng variant.

### 12.4. Snapshot được khóa

| Thành phần | Commit |
|---|---|
| [FOA](https://github.com/jiaxiaojunQAQ/FOA-Attack/tree/6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b) | `6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b` |
| [MPC](https://github.com/LiYuanBoJNU/MPCAttack/tree/7b195a5bcbaeaf1e8534d101199810a2782caad3) | `7b195a5bcbaeaf1e8534d101199810a2782caad3` |
| [K-means](https://github.com/subhadarship/kmeans_pytorch/tree/f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805) | `f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805` |

SHA-256 của file notebook được đọc khi viết report:

```text
868fac0e036bc4e1c4210e5d7af2da493c4df6fae4d2b0598a87b19d539a4002
```

Checksum này chỉ định danh snapshot local của report; notebook được chỉnh sau đó sẽ có checksum khác. HF model IDs ở revision `main` được resolve thành immutable revisions khi khởi tạo run và lưu trong artifacts.

## 13. Cách diễn giải đóng góp sau thực nghiệm

Một đóng góp nghiên cứu có thể được xem xét là **cơ chế phối hợp giữa target-counterfactual discrimination và fresh source-conditioned region sequencing trong một transfer attack có baseline được đối chiếu rõ**. Hiện đây là ứng viên đóng góp, chưa phải novelty đã xác lập.

Để bảo vệ lập luận đó, cần đồng thời có:

1. Baseline reference và runtime thực tế được kiểm tra, với precision/config khác biệt được công bố.
2. H1 được hỗ trợ bởi đối chiếu có/không negative loss và hard/random selection khi bank đủ lớn.
3. H2 được hỗ trợ bởi attention/random-region control; H3 được hỗ trợ bởi fixed/dynamic control cùng phân tích chi phí.
4. Kết quả held-out cải thiện đủ lớn và ổn định, không chỉ surrogate score hoặc một vài ảnh minh họa.
5. Literature review bổ sung chứng minh điểm khác biệt cụ thể so với FOA, MPC, SAGA và các công trình concept/attention-guided transfer khác.

Nếu chỉ một thành phần có lợi, nên thu hẹp phương pháp về thành phần đó. Nếu hybrid không vượt baseline sau các controls, kết quả vẫn giúp xác định giới hạn của attention hoặc counterfactual guidance trong setting này; không cần diễn giải mọi negative result thành lỗi triển khai.

## 14. Nguồn đối chiếu

- Implementation và metadata chính: [notebook v4.3](../../notebooks/attacks/kltn-attack-v4.3-h1.ipynb).
- Tiền thân của attention/crop protocol: [notebook v3.2](../../notebooks/attacks/kltn-attack-v3.2.ipynb).
- Thiết kế cũ để đối chiếu thay đổi: [hypo_v1.md](hypo_v1.md).
- Baseline FOA: [author repository](https://github.com/jiaxiaojunQAQ/FOA-Attack), [pinned loss/extractor source](https://github.com/jiaxiaojunQAQ/FOA-Attack/blob/6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b/surrogates/FeatureExtractors/Base.py).
- Baseline MPC: [paper](https://arxiv.org/abs/2603.04846), [pinned implementation](https://github.com/LiYuanBoJNU/MPCAttack/blob/7b195a5bcbaeaf1e8534d101199810a2782caad3/generate_adversarial_examples_MPCAttack.py), [contrastive score](https://github.com/LiYuanBoJNU/MPCAttack/blob/7b195a5bcbaeaf1e8534d101199810a2782caad3/utils.py).
- Attention region sequencing: [SAGA paper](https://arxiv.org/abs/2602.04356).
- Protocol evaluation local: [v2](../../notebooks/text_gen_eval_v2.ipynb), [v3](../../notebooks/text_gen_eval_v3.ipynb). Report chưa chứng nhận full compatibility của v3 ngoài các cấu hình evaluation đã đọc.
