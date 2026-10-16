# Các giả thuyết nghiên cứu — v1

Các giả thuyết này là **đề xuất cần kiểm chứng**, chưa phải kết luận rằng phương pháp sẽ tăng ASR hoặc đã có novelty.

Tài liệu gồm hai phần: phần đầu giải thích mục tiêu và cách kiểm chứng H1–H4; phần **Đặc tả thực thi toàn bộ pipeline** phía dưới mô tả chính xác cách notebook hiện tại thực hiện chúng, bao gồm sinh negatives, kiểm tra dữ liệu, grounding, memory, loss, optimizer và cấu hình ablation.

## Bối cảnh và mục tiêu chung

Nghiên cứu xét targeted transferable adversarial attack: từ ảnh source, tạo ảnh adversarial có nhiễu nhỏ sao cho mô hình VLM black-box mô tả ảnh adversarial gần với nội dung của ảnh target.

Trong giai đoạn tạo nhiễu:

- InternVL3-1B sinh caption cho ảnh source và target, tương tự implementation của MPCAttack.
- Các CLIP được chọn trong `BACKBONES` đóng vai trò surrogate để tính embedding ảnh, embedding text và gradient. File hiện tại chọn `['B16', 'B32', 'Laion']`, grounding dùng `B16`; có thể chọn riêng `['B16']`.
- Các model được đóng băng; chỉ cập nhật pixels của ảnh adversarial.
- Không sử dụng gradient, attention hoặc phản hồi của victim black-box trong vòng tối ưu.

Ký hiệu:

- $x_s$: ảnh source.
- $x_t$: ảnh target.
- $x_a=x_s+\delta$: ảnh adversarial.
- $c_s,c_t$: caption source và target.
- $f_I,f_T$: image encoder và text encoder của CLIP.
- Các embedding được chuẩn hóa trước khi tính cosine similarity.
- Ràng buộc mặc định: $\|\delta\|_\infty\le16/255$, ảnh nằm trong miền $[0,1]$.

## Giả thuyết 1 — Counterfactual hard negatives giúp tối ưu sự khác biệt giữa target concept và các concept dễ nhầm

### Phát biểu giả thuyết

Nếu ngoài việc kéo embedding ảnh adversarial về phía target, quá trình tối ưu còn buộc ảnh adversarial phân biệt target caption với những caption chỉ khác một concept quan trọng, thì perturbation có thể tập trung hơn vào thông tin phân biệt concept. Điều này có thể cải thiện khả năng transfer sang VLM black-box.

### Vấn đề cần giải quyết

Việc tăng similarity giữa ảnh adversarial và target caption có thể khai thác những thông tin chung như bối cảnh, màu sắc hoặc hành động. Similarity cao chưa bảo đảm model đã tiếp nhận đúng subject hoặc object mong muốn.

Ví dụ:

```text
Source caption:
A dog is running on grass.

Target caption:
A cat is sleeping on a sofa.
```

Ảnh adversarial có thể tăng similarity với target nhờ các thông tin chung về động vật, nhưng vẫn chưa đủ để victim chuyển sang mô tả “cat”.

### Cơ chế đề xuất

Từ target caption, tạo các counterfactual captions bằng cách thay đúng một subject hoặc noun phrase:

```text
Positive:
A cat is sleeping on a sofa.

Negative 1:
A tiger is sleeping on a sofa.

Negative 2:
A rabbit is sleeping on a sofa.

Negative 3:
A fox is sleeping on a sofa.
```

Những caption này giữ nguyên phần lớn ngữ cảnh và thay concept cần phân biệt.

Trong notebook, negative được định nghĩa **so với target mong muốn**. Vì vậy:

- Target caption là positive cần kéo ảnh adversarial tới.
- Counterfactual captions là negatives cần giảm mức nhầm lẫn.
- Source representation là một negative riêng trong joint loss của H4.

Ngân hàng negative captions được sinh một lần cho mỗi cặp ảnh. Trong quá trình attack, chọn lại top-K negatives có cosine similarity cao nhất với embedding ảnh adversarial hiện tại. Đây là các **hard negatives theo trạng thái hiện tại của ảnh**.

### Hàm loss

Đặt $z_a=f_I(x_a)$, $t_t=f_T(c_t)$, và $n_k=f_T(c_k^-)$:

$$
L_H=
-\log
\frac{\exp(\operatorname{sim}(z_a,t_t)/\tau)}
{\exp(\operatorname{sim}(z_a,t_t)/\tau)
+\sum_{k=1}^{K}\exp(\operatorname{sim}(z_a,n_k)/\tau)}
$$

Minimize loss này khuyến khích ảnh adversarial ưu tiên target caption hơn các counterfactual captions.

### Thực nghiệm để kiểm chứng

So sánh:

- `h1_full`: sử dụng hard-negative loss.
- `h1_no_hard_loss`: tắt riêng hard-negative loss, vẫn giữ negatives cho grounding.
- `h1_random_neg`: chọn ngẫu nhiên K negatives từ cùng ngân hàng.
- `h1_no_hard`: bỏ hard-negative loss và bỏ các map riêng của active negatives; source-concept maps vẫn được dùng. **Trong code hiện tại, toàn bộ negative bank vẫn tham gia stationary margin của memory**; đây chưa phải ablation loại negatives hoàn toàn khỏi mọi cơ chế.

`h1_full` so với `h1_no_hard_loss` giúp tách ảnh hưởng của loss khỏi ảnh hưởng của negative-conditioned grounding.

### Điều kiện hỗ trợ giả thuyết

Hard negatives phải cải thiện ASR trên held-out victims so với random negatives hoặc không có hard-negative loss. Việc giảm surrogate loss chưa đủ để kết luận tăng transferability.

### Giới hạn

Caption sửa subject chưa chắc sai hoàn toàn về mặt hình ảnh hoặc đủ “hard”. Cần kiểm tra chất lượng ngân hàng negatives; CLIP filtering không thay thế việc kiểm tra semantic correctness.

**Tiền lệ liên quan:** [CF-VLM](https://arxiv.org/abs/2506.17267), [SADCA](https://arxiv.org/abs/2603.04839v3), [PDCL-Attack](https://arxiv.org/abs/2407.20657). Không coi counterfactual samples hoặc contrastive transfer riêng lẻ là novelty.

## Giả thuyết 2 — Grounding theo concept margin giúp chọn vùng cập nhật nhiễu hiệu quả hơn random regions

### Phát biểu giả thuyết

Nếu vùng cập nhật perturbation được chọn từ attribution của những concept đang cạnh tranh với target concept, thì cùng một diện tích cập nhật có thể tác động hiệu quả hơn vào thông tin semantic cần thay đổi.

### Vấn đề cần giải quyết

Random crop hoặc random region có thể chọn vùng background ít liên quan đến subject cần tấn công. Vòng tối ưu vẫn tiêu tốn gradient updates nhưng chưa tác động đủ vào evidence quyết định nội dung caption.

### Cơ chế đề xuất

Từ source caption, trích xuất những noun phrases quan trọng, ví dụ:

```text
Source concept:
a dog

Desired target:
A cat is sleeping on a sofa.
```

Tính score phân biệt:

$$
s_{\text{source}}(x_a)
=
\operatorname{sim}(f_I(x_a),f_T(\text{source concept}))
-
\operatorname{sim}(f_I(x_a),f_T(c_t))
$$

Với mỗi hard-negative caption đang được chọn:

$$
s_{\text{negative},k}(x_a)
=
\operatorname{sim}(f_I(x_a),f_T(c_k^-))
-
\operatorname{sim}(f_I(x_a),f_T(c_t))
$$

Các score này biểu diễn mức ảnh hiện tại còn ưu tiên source concept hoặc counterfactual caption hơn target caption.

Sau đó dùng gradient của score đối với self-attention trong vision encoder CLIP để tạo relevance maps. Các maps được kết hợp, rồi chọn window có tổng relevance cao nhất.

Trong notebook:

- Vùng cập nhật chiếm khoảng 50% diện tích ảnh.
- Loss được tính trên cả ảnh đầy đủ và crop.
- Gradient cập nhật pixels được giới hạn trong vùng đã chọn.

### Làm rõ khái niệm attention

CLIP là dual encoder, **không có cross-attention trực tiếp từ token chữ sang patch ảnh**.

Vì vậy, notebook sử dụng:

```text
Concept text
→ CLIP text embedding
→ image–text similarity hoặc concept margin
→ gradient đối với vision self-attention
→ text-conditioned relevance map
```

Đây là attribution được conditioning bởi concept text. Không nên mô tả nó là attention trực tiếp của token LLM tới pixels. Phần token spans trong notebook dùng để kiểm tra tokenization, không phải cross-attention weights.

### Thực nghiệm để kiểm chứng

So sánh:

- `h1_random`: random region.
- `h1_fixed`: vùng lấy từ relevance map ban đầu.
- `h1_dynamic`: vùng lấy từ relevance map được cập nhật.

Giữ cùng diện tích vùng, số bước, loss, backbone và ngân hàng negatives. Random control cũng giữ một box trong mỗi block để hạn chế khác biệt về lịch cập nhật.

### Điều kiện hỗ trợ giả thuyết

Grounded regions phải cải thiện ASR so với random regions trong điều kiện kiểm soát tương đương. Heatmap nhìn hợp lý chưa đủ để chứng minh vùng đó giúp attack transfer tốt hơn.

**Tiền lệ liên quan:** [Chefer et al.](https://arxiv.org/abs/2103.15679), [Grad-ECLIP](https://proceedings.mlr.press/v235/zhao24p.html), [Finer-CAM](https://arxiv.org/abs/2501.11309), [GDA](https://arxiv.org/abs/2602.09431v2). GDA đã có text-grounded perturbation allocation; H2 theo nghĩa rộng không phải ý tưởng mới.

## Giả thuyết 3 — Re-grounding trên ảnh adversarial hiện tại, kết hợp memory, giúp phân bổ các lần cập nhật tiếp theo tốt hơn fixed saliency

### Phát biểu giả thuyết

Khi perturbation làm thay đổi representation của ảnh, vùng quan trọng đối với các concept cũng có thể thay đổi. Tính lại relevance trên ảnh adversarial hiện tại và sử dụng lịch sử cập nhật có thể giúp lựa chọn vùng tối ưu tiếp theo hiệu quả hơn việc giữ nguyên saliency map ban đầu.

### Vấn đề cần giải quyết

Một map tính từ ảnh sạch chỉ phản ánh trạng thái ban đầu. Sau nhiều bước attack:

- Evidence của source concept có thể đã suy yếu.
- Concept gây nhầm lẫn có thể thay đổi.
- Một vùng khác có thể trở thành vùng cần ưu tiên.

Tiếp tục dùng map cũ có thể dẫn đến cập nhật lặp lại tại vùng đã mang lại ít lợi ích bổ sung.

### Cơ chế đề xuất

Chia vòng attack thành các block. Mặc định notebook refresh mỗi 30 bước:

```text
Ảnh ban đầu
→ tính relevance
→ chọn vùng
→ cập nhật 30 bước
→ lấy ảnh adversarial hiện tại
→ tính lại relevance
→ chọn vùng tiếp theo
→ tiếp tục cập nhật
```

“Chuỗi ảnh” ở đây là chuỗi trạng thái $x_a^{(0)},x_a^{(30)},x_a^{(60)},\ldots$, không phải video và không phải huấn luyện lại model.

Notebook triển khai hai mức:

**Mức 1: Fresh re-grounding — `h1_dynamic`**

Tính lại map từ ảnh hiện tại và dùng trực tiếp map mới để chọn vùng.

**Mức 2: Re-grounding có memory — `h1_full`**

Kết hợp map mới với map lịch sử bằng EMA:

$$
E_t=\beta E_{t-1}+(1-\beta)R_t
$$

Trong đó $R_t$ là relevance map mới.

Đồng thời theo dõi một surrogate margin trên ảnh đầy đủ:

$$
m_t=
\max_{u\in\mathcal U}
\operatorname{sim}(f_I(x_a^{(t)}),u)
-
\operatorname{sim}(f_I(x_a^{(t)}),f_T(c_t))
$$

$\mathcal U$ chứa source concepts và negative captions. Margin giảm được xem là tiến bộ theo thước đo surrogate này.

Nếu một block làm margin giảm, notebook ghi credit cho các vùng đã được cập nhật trong block. Priority của vùng đã nhận credit được giảm tương đối, tạo cơ hội chuyển sang vùng khác.

### Lưu ý về memory

Credit là heuristic phân bổ lịch sử. Nó không chứng minh vùng nào là nguyên nhân gây cải thiện.

Ngoài ra, đổi vùng cập nhật không xóa perturbation đã tích lũy ở vùng trước. Mask chỉ quyết định pixels được cập nhật ở bước tiếp theo.

### Thực nghiệm để kiểm chứng

So sánh:

- `h1_fixed`: map ban đầu cố định.
- `h1_dynamic`: tính lại map, không memory.
- `h1_full`: tính lại map, có memory.

Cần kiểm soát thêm compute vì re-grounding tốn forward/backward. Ngoài equal-step comparison, cần equal-time hoặc equal-compute comparison.

Cũng cần theo dõi tổng vùng đã từng được cập nhật: cùng diện tích mỗi bước chưa bảo đảm cùng cumulative spatial coverage.

### Điều kiện hỗ trợ giả thuyết

- `h1_dynamic` vượt `h1_fixed`: hỗ trợ lợi ích của re-grounding.
- `h1_full` vượt `h1_dynamic`: hỗ trợ lợi ích bổ sung của memory.

Các kết luận này cần dựa trên transfer ASR sau khi kiểm soát chi phí tính toán.

**Tiền lệ liên quan:** [SAGA](https://arxiv.org/abs/2602.04356v2) đã nghiên cứu region sequencing với fixed attention map; [SADCA](https://arxiv.org/abs/2603.04839v3) đã có dynamic image/text interactions. Không coi “dynamic update” hoặc “region sequence” riêng lẻ là novelty.

## Giả thuyết 4 — Fused image/text anchors giúp hướng perturbation về target semantics tốt hơn image-only anchors

### Phát biểu giả thuyết

Nếu source và target được biểu diễn bằng cả image embedding và caption embedding trong không gian CLIP, thì quá trình contrastive optimization có thể được hướng dẫn rõ hơn về nội dung semantic cần rời khỏi và cần tiến tới.

### Vấn đề cần giải quyết

Image embedding chứa cả subject, background, texture và những thông tin khác. Caption bổ sung một biểu diễn ngôn ngữ về nội dung ảnh, nhưng cũng có thể mất chi tiết hoặc hallucinate.

Giả thuyết cần kiểm tra là việc kết hợp hai loại thông tin có giúp targeted transfer tốt hơn hay không.

### Cơ chế đề xuất

Tạo joint anchors:

$$
j_s=
\operatorname{norm}
\left(
\lambda f_I(x_s)+(1-\lambda)f_T(c_s)
\right)
$$

$$
j_t=
\operatorname{norm}
\left(
\lambda f_I(x_t)+(1-\lambda)f_T(c_t)
\right)
$$

Mặc định $\lambda=0.6$.

Embedding ảnh adversarial là:

$$
z_a=f_I(x_a)
$$

Ảnh adversarial được tối ưu để ưu tiên target anchor hơn source anchor:

$$
L_J=
-\log
\frac{\exp(\omega\operatorname{sim}(z_a,j_t)/\tau)}
{\exp(\omega\operatorname{sim}(z_a,j_t)/\tau)
+\exp(\operatorname{sim}(z_a,j_s)/\tau)}
$$

Mặc định $\omega=2,\tau=0.2$.

Source/target captions và anchors được giữ cố định trong vòng attack. Không sinh lại caption adversarial ở mỗi bước và không backprop qua caption generator.

### Thực nghiệm để kiểm chứng

So sánh:

- `h1_full`: joint anchors có image/text fusion.
- `h1_image_anchors`: giữ joint contrastive loss nhưng anchors chỉ dùng image embeddings.

Đây là đối chiếu trực tiếp cho lợi ích của fusion.

`h1_no_fusion` trong notebook bỏ toàn bộ joint loss, nên chỉ đo đóng góp của joint-loss term; nó không tách riêng hiệu quả của image/text fusion.

### Điều kiện hỗ trợ giả thuyết

Fused anchors phải cải thiện ASR so với image-only anchors khi các loss khác, grounding, perturbation budget và backbone được giữ nguyên.

**Tiền lệ liên quan:** [MPCAttack, §3.2](https://arxiv.org/html/2603.04846v1#S3.SS2) đã có caption encoding và image/text fusion. H4 là giả thuyết ablation cho hệ thống đề xuất, không phải đóng góp mới độc lập.

## Giả thuyết tổng hợp của notebook

Việc kết hợp:

1. Counterfactual hard negatives để xác định concept đang cạnh tranh với target.
2. Concept-margin relevance để chọn vùng cập nhật.
3. Re-grounding trên ảnh adversarial hiện tại.
4. Memory để điều chỉnh ưu tiên qua các block.
5. Image/text joint anchors để hướng dẫn targeted optimization.

có thể tạo perturbation transfer hiệu quả hơn các control sử dụng random regions, fixed maps hoặc không có hard-negative loss.

Loss tổng mặc định:

$$
L=L_{\text{FOA-derived}}+L_J+0.5L_H
$$

Trong đó FOA-derived loss gồm global feature alignment và local clustering/OT alignment. Đây là tổ hợp đề xuất, có các điều chỉnh so với implementation gốc của [FOA-Attack](https://arxiv.org/abs/2505.21494); không phải reproduction nguyên bản FOA hoặc MPC.

## Câu hỏi nghiên cứu hẹp để viết trong đề cương

“Trong targeted transferable adversarial attacks sử dụng CLIP làm surrogate, liệu counterfactual hard negatives có giúp xác định vùng semantic cần cập nhật, và liệu re-grounding kết hợp memory trên chuỗi ảnh adversarial có cải thiện ASR trên VLM black-box so với fixed saliency và random regions dưới cùng perturbation budget hay không?”

Đây là câu hỏi về hiệu quả của cơ chế kết hợp. **Novelty của cơ chế này vẫn cần rà soát thêm; tăng surrogate similarity hoặc có heatmap đẹp chưa chứng minh tăng black-box ASR.**

## Đặc tả thực thi toàn bộ pipeline

Phần này mô tả implementation của notebook, không gán những lựa chọn đề xuất cho paper FOA hoặc MPC. Source captions, target captions và ngân hàng negatives là các đối tượng khác nhau; phân biệt chúng là điều cần thiết để hiểu chiều tối ưu.

### 1. Vai trò của từng model và từng loại representation

| Thành phần                | Model trong code                        | Chức năng                                                                                        | Có cập nhật weights không?       |
| ------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------- |
| Caption/concept generator | `OpenGVLab/InternVL3-1B-hf`             | Nhìn source/target để sinh captions; sau đó nhận text để đề xuất source concepts và target edits | Không                            |
| CLIP B16                  | `openai/clip-vit-base-patch16`          | Image/text embeddings; hidden global/patch features; grounding mặc định                          | Không                            |
| CLIP B32                  | `openai/clip-vit-base-patch32`          | Image/text embeddings và hidden features nếu được chọn                                           | Không                            |
| CLIP Laion                | `laion/CLIP-ViT-G-14-laion2B-s12B-b42K` | Image/text embeddings và hidden features nếu được chọn                                           | Không                            |
| Victim black-box          | Chọn ở giai đoạn evaluation riêng       | Sinh mô tả ảnh để đo transfer                                                                    | Không được gọi trong vòng attack |

CLIP có hai nhóm features được dùng cho hai mục đích:

- **Projected image/text embeddings:** `visual_projection(pooler_output)` và `text_projection(pooler_output)`, sau đó chuẩn hóa L2. Đây là các vectors cùng không gian để tính joint loss, hard-negative loss, negative ranking và grounding score.
- **Hidden vision features:** CLS token và patch tokens từ `last_hidden_state`, chuẩn hóa L2. Đây là đầu vào của FOA-derived global/local loss. Không trộn hidden patch vector trực tiếp với text embedding.

Mỗi backbone mã hóa text bằng **text tower riêng của chính CLIP đó**. Image/text fusion được thực hiện trong từng backbone; vectors khác dimension từ B16/B32/Laion không được cộng trực tiếp với nhau. Gradients đối với pixels được tổng hợp sau khi tính loss riêng từng backbone.

Text towers được giữ trên CPU; embeddings được cache rồi chuyển sang attack device. Vision towers và visual projections nằm trên attack device. InternVL được giải phóng sau khi hoàn tất giai đoạn captions/concepts cho các cặp đã chọn, trước khi tải CLIP cho attack.

Không dùng InternVL hidden text embeddings hoặc InternVL/DINO image features trong loss. Đây là phiên bản CLIP-only lấy cảm hứng từ caption/fusion/contrastive optimization của [MPC, §3.2 Eq. 3–6](https://arxiv.org/html/2603.04846v1#S3.SS2), không phải toàn bộ multi-paradigm method gốc.

### 2. Chuẩn bị dữ liệu và sinh source/target captions

1. `prepare_pairs()` tìm thư mục `bigscale` và `target_images`, hoặc dùng `SOURCE_DIR`/`TARGET_DIR` do người chạy chỉ định.
2. Mặc định ghép source/target theo cùng filename stem, ví dụ `0.png` với `0.jpg`. Stem trùng lặp trong một phía hoặc hai phía không khớp sẽ gây lỗi. Thứ tự stem là sort dạng chuỗi; không tự chuyển thành sort số.
3. Nếu cần giữ pairing của baseline, dùng `PAIR_MANIFEST`; code lấy stems từ `original_path` và `target_path`, giữ `index`/`sample_id` của manifest.
4. Resize bicubic với cạnh ngắn 224, center crop 224×224, chuyển RGB, tạo tensor `[1,3,224,224]` trong `[0,1]`.
5. Ghi raw-file hash và processed-pixel hash để phát hiện dữ liệu thay đổi.
6. InternVL nhận **ảnh đã preprocess**, với prompt:

```text
Describe this image in one concise sentence, no longer than 20 words.
```

`generate()` dùng `do_sample=False`. `CAPTION_MAX_NEW_TOKENS=80` là giới hạn sinh token, khác giới hạn 20 từ trong prompt. Code không đếm và cưỡng chế số từ; nó kiểm tra caption không rỗng và báo lỗi nếu phần sinh chạm token cap. Nếu caption/concept text vượt context length của CLIP khi encode, code cũng báo lỗi thay vì âm thầm truncate.

Hai captions được giữ cố định cho toàn bộ vòng attack của một mẫu. Các variants trong cùng run dùng chung captions và candidate bank. `MANUAL_TEXT_OVERRIDES` cho phép cung cấp captions và/hoặc concepts đã chuẩn bị, nhưng concepts vẫn phải qua parser.

### 3. Negative samples thực sự được sinh ra như thế nào?

**Negative samples trong H1 là các câu text được sửa concept từ target caption. Không sinh negative images, không lấy ảnh bất kỳ từ dataset, và không dùng CLIP để sinh câu.**

Chuỗi xử lý là:

```text
Source/target images
→ InternVL sinh hai captions
→ InternVL nhận hai captions, đề xuất source_concepts và target_edits
→ Python thay từng span trong target caption để tạo candidate negative captions
→ parser kiểm tra edits
→ CLIP lọc candidates
→ giữ một bank cố định
→ chọn lại hard top-K từ bank theo ảnh adversarial hiện tại
```

#### 3.1. InternVL chỉ đề xuất edits trong lượt sinh concept

Sau khi có captions, `MPCCaptioner.concepts(source, target, sample_id)` gọi lại **cùng InternVL3-1B**, nhưng lần này `image=None`: đầu vào chỉ là text. Model không nhìn lại ảnh target khi đề xuất replacements.

Với cấu hình hiện tại, nội dung yêu cầu tương ứng là:

```text
Source caption: <source caption>
Target caption: <target caption>
Return ONLY JSON with keys source_concepts and target_edits.
source_concepts: up to 3 important noun phrases copied exactly from the SOURCE caption.
target_edits: 6 objects with span and replacement.
Each span is one short subject/object noun phrase copied exactly from the TARGET caption, occurring once.
Replace it with a different plausible but incorrect object/category,
not a synonym, not a pronoun, not a negation.
Keep all surrounding words unchanged. Do not replace the entire sentence.
```

Prompt thực tế trong code còn kèm ví dụ JSON và yêu cầu dùng captions đầu vào, không sao chép động vật từ ví dụ. `CONCEPT_MAX_NEW_TOKENS=512`, `do_sample=False`.

Ví dụ dưới đây chỉ minh họa định dạng, không phải kết quả đã chạy model:

```json
{
  "source_concepts": ["A dog"],
  "target_edits": [
    { "span": "A cat", "replacement": "A tiger" },
    { "span": "A cat", "replacement": "A rabbit" },
    { "span": "A cat", "replacement": "A fox" }
  ]
}
```

Với target `A cat is sleeping on a sofa.`, Python tạo lần lượt:

```text
A tiger is sleeping on a sofa.
A rabbit is sleeping on a sofa.
A fox is sleeping on a sofa.
```

Các edits được áp dụng **độc lập lên caption target gốc**, không áp dụng nối tiếp lên câu đã sửa. Vì vậy mỗi candidate chỉ có một span được thay. Edits có thể nhắm vào những subject/object spans khác nhau; code không bắt tất cả candidates phải sửa cùng một subject.

#### 3.2. Parser kiểm tra những gì?

`parse_concepts()` kiểm tra:

| Đối tượng       | Kiểm tra thực sự trong code                                                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JSON            | Lấy nội dung từ dấu `{` đầu tiên đến `}` cuối cùng, parse thành object; phải có hai lists `source_concepts` và `target_edits`                             |
| Source concepts | Chuỗi không rỗng, xuất hiện trong source caption theo so khớp không phân biệt hoa/thường; loại trùng sau normalize; cần ít nhất một concept, giữ tối đa 3 |
| Edit            | Có `span` và `replacement` là chuỗi không rỗng                                                                                                            |
| Span            | Khớp đúng một lần trong target caption, có word boundaries; không quá 6 từ; không bằng toàn bộ caption sau normalize                                      |
| Replacement     | Không bằng span sau normalize                                                                                                                             |
| Câu negative    | Không trùng source caption, target caption hoặc negative đã giữ sau normalize                                                                             |
| Số lượng        | Cần ít nhất `MIN_NEGATIVES=2` candidates hợp lệ; giữ tối đa `MAX_NEGATIVES=6` theo thứ tự sinh                                                            |

Normalize ở đây là chuyển casefold, thay các nhóm ký tự không phải word bằng khoảng trắng rồi strip. Parser dùng regex có boundaries để tránh thay `cat` bên trong `cattle`.

**Prompt yêu cầu noun phrases, không dùng synonym/pronoun/negation; parser không có bộ kiểm tra ngôn ngữ để chứng nhận những điều đó.** Parser cũng không chứng minh replacement là object category hợp lý hoặc sai với ảnh. Cần phân biệt yêu cầu đưa cho generator với điều kiện được kiểm tra bằng code.

Parser chấp nhận số edits nhỏ hơn 6 nếu vẫn còn ít nhất 2 candidates hợp lệ. Nó không bắt model trả đúng 6 candidates và không xếp hạng hardness ở giai đoạn này.

#### 3.3. Retry và xử lý thất bại

Nếu JSON sai, edits không đủ hợp lệ hoặc generation không hợp lệ, `concepts()` thử tối đa `TEXT_RETRIES=3` lần. Prompt của lần sau được nối thêm thông báo lỗi của lần trước và yêu cầu sửa JSON; đây không phải ba lần sampling ngẫu nhiên độc lập.

Nếu hết retries, mẫu báo lỗi và có thể được cung cấp lại bằng `MANUAL_TEXT_OVERRIDES`. Code không tự bịa một fallback bank. `CONTINUE_ON_SAMPLE_ERROR=True` cho phép tiếp tục các mẫu khác đối với lỗi thông thường; lỗi thiếu GPU memory hoặc người dùng interrupt làm dừng giai đoạn tương ứng. Source/target caption generation không có vòng retry ba lần giống concept generation.

Raw JSON hợp lệ cuối cùng, rejected edits và candidates được lưu để audit. Các lần sinh concept lỗi được ghi event với lý do; không có cam kết lưu toàn bộ raw response của mọi lần lỗi.

#### 3.4. Lọc candidate bank bằng CLIP

Trong `build_anchors()`, encode captions, source concepts và candidate negative captions bằng mỗi CLIP. Việc quyết định giữ candidate được thực hiện **chỉ bằng `GROUNDING_BACKBONE`**, hiện là B16.

Đặt $t_t=f_T(c_t)$, $n_j=f_T(c_j^-)$, $z_t=f_I(x_t)$ của backbone grounding. Candidate được giữ khi đồng thời thỏa:

$$
\operatorname{sim}(n_j,t_t)<0.98
$$

và

$$
\operatorname{sim}(z_t,n_j)
\le
\operatorname{sim}(z_t,t_t)-\Delta,
\qquad \Delta=\texttt{NEGATIVE\_TARGET\_MARGIN}=0.
$$

Điều kiện đầu loại text embeddings gần như trùng positive theo ngưỡng đã chọn. Điều kiện sau loại negative mà CLIP đánh giá còn hợp với target image hơn positive caption. Với $\Delta=0$, bằng điểm vẫn được giữ; nếu tăng $\Delta$ thì yêu cầu khoảng cách rõ hơn.

Đây là heuristic lọc, **không chứng nhận semantic falsehood**. Code lưu `semantic_falsehood_verified=False`. Không có thêm vòng hỏi InternVL/victim xác minh lại candidate theo ảnh.

Nếu sau filter còn dưới 2 negatives, `build_anchors()` báo lỗi. Không tự hạ threshold, sinh thêm negatives hoặc giữ lại candidate đã bị loại. Những indices được B16 giữ được áp dụng đồng nhất cho bank của các CLIP khác để cùng so sánh một tập câu.

#### 3.5. Chọn hard negatives trong vòng attack

Bank sau filter được giữ cố định. Tại steps 0, 10, 20, …, `mine_negatives()` encode **ảnh adversarial đầy đủ hiện tại**, không phải crop. Với candidate $j$:

$$
q_j^{(t)}=\frac{1}{M}\sum_{m=1}^{M}
\operatorname{sim}\left(f_{I,m}(x_a^{(t)}),f_{T,m}(c_j^-)\right).
$$

Chọn top `HARD_TOP_K=2` candidates có $q_j$ lớn nhất. Điểm ranking là trung bình đều các CLIP; **không dùng dynamic ensemble weights của attack loss**. Ranking không cộng source concepts vào candidate bank và không dùng victim feedback.

Ví dụ minh họa, không phải số đo thực nghiệm:

| Candidate      | Similarity trung bình với ảnh adversarial hiện tại | Lựa chọn                   |
| -------------- | -------------------------------------------------: | -------------------------- |
| tiger caption  |                                               0.31 | Top-2                      |
| rabbit caption |                                               0.37 | Top-1                      |
| fox caption    |                                               0.24 | Chưa chọn ở lần mining này |

Tới lần mining sau, điểm và top-K có thể đổi. Với `negatives='random'`, code chọn K indices bằng `randperm` từ cùng bank, không replacement trong một lượt; với `negatives='none'`, trả về lists rỗng và không chạy ranking forwards.

Không backprop qua top-K, text generation hoặc bank filtering. Embeddings của text và bank không được tối ưu; chỉ ảnh thay đổi.

#### 3.6. Phân biệt các loại “negative”

| Đối tượng                               | Negative so với cái gì?                      | Dùng ở đâu?                                                                |
| --------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------- |
| Source joint anchor $j_s$               | Target joint anchor mong muốn                | Denominator của $L_J$                                                      |
| Counterfactual negative caption $c_j^-$ | Target caption $c_t$                         | Bank, top-K, $L_H$, negative-conditioned maps                              |
| Source concept                          | Concept đích mong muốn trong grounding score | Source-concept maps và stationary margin; không tự thành phần tử của $L_H$ |

Trong $L_H$, model được kéo **tới target caption và ra xa counterfactual captions**. Không gọi counterfactual là “negative” rồi kéo ảnh về nó. Đây cũng là lý do không thể lấy nguyên quy ước positive/negative của một untargeted attack như SADCA rồi áp dụng mà giữ nguyên chiều loss.

### 4. H2 được triển khai: từ concept margins đến vùng cập nhật

`ConceptGrounder.refresh()` dùng một CLIP, hiện là B16. Nó lấy ảnh adversarial hiện tại, detach khỏi lịch sử optimizer rồi bật gradient cho pixels. Models vẫn frozen.

Tập điều kiện tạo maps là:

```text
Các source-concept embeddings
+ Các full negative-caption embeddings được chọn tại lần mining gần nhất
```

Source concept được encode thành **phrase riêng**, không lấy contextual token embedding bên trong caption dài. Active negative được encode thành **cả caption đã sửa**, không chỉ replacement noun. Đây là hai loại input text khác nhau về độ dài/ngữ cảnh; cần ghi rõ khi phân tích attribution.

Với mỗi vector điều kiện $u_i$:

$$
s_i=\operatorname{sim}(z_a,u_i)-\operatorname{sim}(z_a,t_t).
$$

Score cao nghĩa là ảnh hiện tại còn ưu tiên điều kiện đó tương đối so với target caption. Notebook dùng score này làm attribution target, không dùng toàn bộ attack loss làm attribution target.

#### 4.1. Gradient-weighted attention rollout

Lấy `GROUNDING_LAST_LAYERS=6` self-attention layers cuối của vision tower. Với attention $A_\ell$ và gradient của score $s_i$:

$$
\bar A_{\ell,i}
=\operatorname{mean}_{\text{heads}}
\left[\operatorname{ReLU}\left(A_\ell\odot\frac{\partial s_i}{\partial A_\ell}\right)\right].
$$

Khởi tạo $R=I$ và truyền lần lượt theo thứ tự layers:

$$
R\leftarrow R+\bar A_{\ell,i}R.
$$

Lấy hàng CLS tới các patch tokens, reshape thành patch grid, nội suy bilinear về 224×224, rồi chuẩn hóa tổng bằng 1. Với B16 ở 224×224, patch grid là 14×14.

Mỗi concept/negative có map riêng. Map tổng dùng weights:

$$
\pi_i=\operatorname{softmax}_i(s_i/\tau),\qquad
R_{\text{fresh}}=\operatorname{Normalize}\left(\sum_i\pi_iR_i\right).
$$

Maps và weights được detach trước khi chọn vùng; không lấy second-order gradients qua quá trình grounding. Đây là implementation gradient-weighted rollout lấy cảm hứng từ [Chefer et al.](https://arxiv.org/abs/2103.15679), không phải implementation Grad-ECLIP hoặc Finer-CAM. [Grad-ECLIP](https://proceedings.mlr.press/v235/zhao24p.html) và [Finer-CAM](https://arxiv.org/abs/2501.11309) là related work/comparators, chưa có variant tái tạo riêng trong notebook.

#### 4.2. Fallback và chọn window

`probability_map()` thay giá trị NaN/Inf bằng 0, clamp âm về 0. Nếu tổng map không đủ lớn, trả về uniform map và ghi số fallback. Nếu attention tensors không có hoặc gradient attention không nối với score, code báo lỗi; không âm thầm lấy lại map cũ.

`choose_box()` trượt window hình vuông trên priority map, chọn window có mean relevance lớn nhất. Do window có cùng kích thước ở mọi vị trí, cách này tương đương tối đa hóa tổng relevance. Nếu đồng hạng, `argmax` lấy vị trí đầu theo thứ tự flatten; uniform map vì vậy có thể chọn box góc trên trái, **không tự biến thành random crop**.

Với `CROP_AREA=0.5`, cạnh window được tính bằng `round(sqrt(0.5*224*224))=158`. Diện tích thực là 158×158, xấp xỉ 49.75% ảnh, không chính xác 50% do làm tròn.

Box tạo ra hai đối tượng:

- `crop_view`: cắt vùng từ ảnh adversarial rồi resize bilinear về 224×224 để tính local view loss.
- `support_mask`: mask nhị phân tại vị trí box trong ảnh gốc để giới hạn pixel update.

CLIP attention map không phải segmentation ground truth. Token spans được lưu chỉ để audit tokenizer, không dùng làm cross-attention text-to-patch.

### 5. H3 được triển khai: lịch refresh, EMA và memory credit

#### 5.1. Hai lịch cập nhật độc lập

Với 300 updates và mặc định hiện tại:

| Cơ chế            | Steps nội bộ, bắt đầu từ 0 | Đối tượng được đổi                          |
| ----------------- | -------------------------- | ------------------------------------------- |
| Negative mining   | 0, 10, 20, …, 290          | Active top-K indices và ranking scores      |
| Dynamic grounding | 0, 30, 60, …, 270          | Concept maps, memory state, priority và box |
| Fixed grounding   | Chỉ step 0                 | Một map/box dùng cho cả 300 updates         |
| Random region     | 0, 30, 60, …, 270          | Box random mới, giữ nguyên trong block      |

Tại step có cả hai cơ chế, mining chạy **trước** grounding. Map tại step 30 được tính sau 30 pixel updates và dùng cho update thứ 31. File trace đánh số update từ 1; metadata map dùng `computed_after_steps` bắt đầu từ 0.

Khi top-K đổi ở step 10 hoặc 20, hard loss dùng negatives mới ngay, nhưng box/map giữ nguyên tới step 30. Vì vậy active negatives trong loss và negatives đã dùng tạo map có thể khác nhau giữa hai lần refresh. Không sinh caption/bank mới theo lịch này.

#### 5.2. Stationary margin để đo tiến bộ của block

Memory dùng một tập tham chiếu cố định cho một mẫu:

$$
\mathcal U=\{\text{mọi source-concept embeddings}\}\cup\{\text{mọi filtered negative-caption embeddings}\}.
$$

Tại mỗi refresh, tính $m_r=\max_{u\in\mathcal U}\operatorname{sim}(z_a,u)-\operatorname{sim}(z_a,t_t)$ trên ảnh đầy đủ của grounding CLIP. Dùng toàn bank thay vì chỉ top-K giúp thước đo không đổi tập tham chiếu chỉ vì mining đổi indices.

Tiến bộ surrogate giữa hai refresh:

$$
p_r=\max(0,m_{r-1}-m_r).
$$

Nếu margin tăng hoặc không giảm, $p_r=0$. Đây không phải ASR, không phải tín hiệu từ victim, và không phải trực tiếp mức giảm của loss tổng full/crop.

#### 5.3. Ghi nhận visits, credit và EMA

Sau mỗi pixel update, `note_region()` cộng 1 vào visits của pixels trong box. Tới refresh tiếp theo, chuẩn hóa visits thành phân phối $V_r$:

$$
V_r=\frac{\mathrm{visits}}{\max(\sum\mathrm{visits},1)}.
$$

Với variant có memory:

$$
C_r=0.9C_{r-1}+p_rV_r,
$$

$$
E_r=0.7E_{r-1}+0.3R_{\mathrm{fresh},r}.
$$

Khởi tạo lần đầu: $C_0=0$, $E_0=R_{\mathrm{fresh},0}$. Chuẩn hóa credit theo cực đại và tạo priority:

$$
\widehat C_r=\frac{C_r}{\max(\max C_r,10^{-12})},\qquad
P_r=\operatorname{Normalize}\left(\frac{E_r}{1+1.0\widehat C_r}\right).
$$

Sau refresh, visits được reset về 0. Credit có decay; EMA giúp tránh chỉ phụ thuộc map mới. Vùng có credit cao bị discount tương đối, nhưng vẫn có thể được chọn lại nếu relevance còn đủ cao. Code không bắt buộc box phải di chuyển.

Không có memory thì priority là fresh map. Với fixed variant, không refresh sau step 0 nên không có chuỗi cập nhật EMA/credit. Trạng thái memory được khởi tạo riêng cho từng mẫu và từng variant.

**Giới hạn:** credit phân bổ theo visits trong block, không theo phép can thiệp đối chứng từng pixel. Nó không xác định causal effect. Thay đổi margin có thể do tương tác toàn ảnh và các updates trước đó.

**Chi tiết ablation dễ bỏ sót:** `h1_no_hard` không tạo maps riêng cho negatives, nhưng `ConceptGrounder.refresh()` vẫn dùng `anchor['negative_texts']` trong stationary margin. `CREDIT_STRENGTH=0` bỏ discount nhưng vẫn còn EMA; `MEMORY_EMA=0` bỏ smoothing nhưng vẫn có credit nếu `memory=True`. Muốn tắt cả hai cơ chế, dùng `memory=False`, như `h1_dynamic`.

### 6. H4 và FOA-derived loss kết hợp như thế nào?

#### 6.1. Ba objective riêng biệt

Joint loss $L_J$ dùng target/source fused anchors và hệ số positive `POSITIVE_OMEGA=2`. Hard loss $L_H$ dùng **target text embedding**, active negative text embeddings và positive weight 1. Target joint anchor và target text embedding không phải cùng một vector.

FOA-derived loss không trực tiếp dùng caption:

$$
L_F=(1-\operatorname{sim}(g_a,g_t))+0.2\left(1-\operatorname{OTSim}(P_a,P_t)\right).
$$

$g$ là hidden CLS features; $P$ là centers của patch clusters. Target full-image features/centers được cache, không crop target lại trong từng step.

K-means dùng 10 clusters, farthest-point initialization xác định và tối đa 15 iterations. Assignments được tính detached; gradients vẫn đi qua cluster means được tạo từ adversarial patch features. Empty cluster dùng patch fallback tương ứng. OT dùng cosine matrix, uniform marginal masses, log-space Sinkhorn 60 iterations với regularization 0.1. Transport plan được detach; gradient đi qua similarity matrix và adversarial centers, không qua các Sinkhorn updates.

Đây là **FOA-derived** implementation với các điều chỉnh đã nêu; không gọi là reproduction chính xác toàn bộ thuật toán tác giả. Việc dùng CLIP như FOA không bắt buộc phải bật $L_F$.

#### 6.2. Full/crop averaging và tổng hợp nhiều CLIP

Mỗi objective được tính trên hai views, rồi trung bình:

$$
\bar L_{q,m}=\frac{L_{q,m}(x_a)+L_{q,m}(\operatorname{crop}(x_a,B))}{2},
\qquad q\in\{F,J,H\}.
$$

Crop và full views dùng cùng active negative IDs và cùng target/source anchors. Không chọn lại hard negatives riêng cho crop. Mỗi CLIP có:

$$
L_m=a\bar L_{F,m}+b\bar L_{J,m}+c\bar L_{H,m},
\qquad (a,b,c)=(1,1,0.5).
$$

Code tính gradient từng CLIP tuần tự để giảm peak graph memory. Bước đầu ensemble weights đều nhau; ở các bước sau, nếu `USE_DYNAMIC_ENSEMBLE=True`:

$$
w_m^{(t)}=\operatorname{softmax}_m\left(\frac{L_m^{(t)}}{\max(L_m^{(t-1)},10^{-8})}\right),\qquad
G^{(t)}=\sum_m w_m^{(t)}\nabla_xL_m^{(t)}.
$$

Weights được detached và reset từng mẫu/variant; không lấy gradient qua phép cập nhật weights. Không có thêm hệ số $1/M$ ngoài weights đã có tổng bằng 1. Chọn một CLIP thì weight bằng 1.

Ratio này dùng losses full/crop tại hai bước liên tiếp; nếu box hoặc active negatives đổi, loss có thể đổi do chính objective/view thay đổi. Vì vậy không diễn giải ratio như thước đo thuần túy tiến bộ của backbone. Nó cũng độc lập với stationary margin dùng cho memory.

FOA hướng dẫn global/local visual matching; joint loss hướng dẫn target-vs-source image/text semantics; hard loss hướng dẫn target-text-vs-counterfactual discrimination. Sự bổ sung giữa chúng là giả thuyết thực nghiệm, không phải bảo đảm rằng cộng thêm loss sẽ tăng transfer.

### 7. Cập nhật pixels, momentum và projection

Khởi tạo $x_a^{(0)}=x_s$, không random-start perturbation trong implementation hiện tại. Source image làm tâm cố định của perturbation ball.

Với support mask $M_B$:

$$
\widetilde G_t=G_t\odot M_B,
$$

$$
v_t=\left(\mu v_{t-1}+\frac{\widetilde G_t}{\max(\operatorname{mean}|\widetilde G_t|,10^{-12})}\right)\odot M_B,
$$

$$
x_a^{(t+1)}=\Pi_{[0,1]\cap[x_s-\epsilon,x_s+\epsilon]}
\left(x_a^{(t)}-\alpha\operatorname{sign}(v_t)\right).
$$

Mặc định $\mu=0$, $\alpha=1/255$, $\epsilon=16/255$, 300 updates. **Dùng dấu trừ vì đang minimize các CE/distance losses.** Không sao chép dấu gradient ascent từ một implementation trả về negative CE rồi giữ nguyên loss dương.

Mask trước và sau momentum ngăn momentum của vùng cũ làm rò updates ra ngoài box hiện tại. Perturbation đã tích lũy ngoài box không bị xóa. Ngân sách $\epsilon$ là theo từng pixel, không phải tổng ngân sách chia đều cho các vùng qua thời gian.

`GRADIENT_SCALE=1024` nhân loss trước backward rồi chia gradient lại, phục vụ numerical scaling; không phải loss weight tăng 1024 lần. Có kiểm tra loss/gradient hữu hạn.

Ảnh cuối được round sang uint8 PNG và clamp lại quanh source pixels với radius `floor(EPSILON_255)`; sau đó đọc PNG để kiểm tra L∞ thực sự trên file. Nếu epsilon là số lẻ nhỏ hơn một pixel, lượng nhiễu khả dụng trên file uint8 có thể bị giảm do lượng tử hóa. Checkpoints/maps preview là ảnh round để quan sát; grounding tối ưu dùng tensor hiện tại, không đọc lại PNG checkpoint vào vòng attack.

### 8. Pseudocode cho toàn bộ một run và một mẫu

```python
# Giai đoạn chuẩn bị: chạy một lần cho các mẫu đã chọn.
pairs = prepare_pairs()
resolve_and_record_model_revisions_and_run_fingerprint()
for pair in pairs:
    source_caption = InternVL(preprocess(pair.source), caption_prompt)
    target_caption = InternVL(preprocess(pair.target), caption_prompt)
    edits_json = InternVL(text_only_prompt(source_caption, target_caption))
    text_record = parse_concepts(edits_json, source_caption, target_caption)
    cache_captions_and_candidate_bank(text_record)
unload_InternVL()
load_selected_CLIP_models()

# Source/target và bank được dùng chung cho các variants của một mẫu.
for pair in pairs:
    anchors, audit = build_anchors_and_filter_negatives(pair, text_record)
    for variant in RUN_VARIANTS:
        image = source.clone()
        reset_momentum_ensemble_history_memory_and_rngs()
        for step in range(STEPS):
            if step % MINE_EVERY == 0:
                active_negatives = mine_full_image(image, fixed_bank, variant)
            if should_refresh_grounding(step, variant):
                fresh_maps = concept_margin_attention_rollout(image, active_negatives)
                priority = apply_memory_if_enabled(fresh_maps)
                box = maximum_mass_window(priority)
            elif is_random_region_refresh(step, variant):
                box = random_window_with_the_same_area()
            for clip in models:
                loss = weighted_average_of_full_and_crop_objectives(clip, image, box)
                image_gradient[clip] = backward_to_pixels(loss)
            gradient = combine_gradients_with_detached_ensemble_weights()
            image = masked_sign_descent_then_project(image, gradient, box, source)
            accumulate_region_visits_and_write_trace()
        save_lossless_images_status_and_sample_record()
rebuild_manifests_summaries_and_export_zip()
```

Pseudocode lược bỏ cache/manual overrides/error branches cho dễ đọc; các nhánh đó có trong code. Không có victim query hoặc model-weight training trong các bước trên.

### 9. Ablations: từng variant thay đổi chính xác điều gì?

| Variant            | FOA loss | Joint loss             | Hard loss / negative selection   | Region                      | Memory                             |
| ------------------ | -------- | ---------------------- | -------------------------------- | --------------------------- | ---------------------------------- |
| `foa_clip_control` | Có       | Không                  | Không / none                     | Random theo block           | Không                              |
| `mpc_clip_control` | Không    | Có                     | Không / none                     | Random theo block           | Không                              |
| `h1_random`        | Có       | Có                     | Có / hard top-K                  | Random theo block           | Không                              |
| `h1_fixed`         | Có       | Có                     | Có / hard top-K                  | Fixed initial map/box       | Không                              |
| `h1_dynamic`       | Có       | Có                     | Có / hard top-K                  | Fresh re-grounding          | Không                              |
| `h1_full`          | Có       | Có                     | Có / hard top-K                  | Re-grounding                | EMA + credit                       |
| `h1_no_hard`       | Có       | Có                     | Không / none                     | Source-concept re-grounding | EMA + credit; margin vẫn chứa bank |
| `h1_random_neg`    | Có       | Có                     | Có / random K                    | Re-grounding                | EMA + credit                       |
| `h1_no_fusion`     | Có       | Không                  | Có / hard top-K                  | Re-grounding                | EMA + credit                       |
| `h1_no_hard_loss`  | Có       | Có                     | Không / hard top-K cho grounding | Re-grounding                | EMA + credit                       |
| `h1_image_anchors` | Có       | Có, image-only anchors | Có / hard top-K                  | Re-grounding                | EMA + credit                       |

`hard_loss` và `fusion` mặc định là `True` nếu không có key tương ứng trong variant. `h1_no_fusion` là tên legacy dễ nhầm: nó bỏ joint loss, không chỉ thay fusion ratio. `h1_image_anchors` mới là control giữ joint loss và đổi anchors từ fused sang image-only.

Những giới hạn cần giữ khi diễn giải kết quả:

- H1: `h1_full` so với `h1_no_hard_loss` đo hiệu quả **thêm hard loss vào cùng cơ chế grounding**, không chứng minh toàn bộ phương pháp sinh bank là mới. Hai variants có thể đi theo trajectories khác sau những updates đầu dù cùng rule grounding.
- H2: `h1_random` so với `h1_dynamic` đồng thời khác content và khả năng thích ứng của region selection. Nếu muốn cô lập text-conditioned map so với non-text attention map, notebook chưa có control đó.
- H3: so sánh fixed/fresh/memory cần kiểm soát compute và cumulative support. Equal steps chưa tương đương equal cost.
- H4: `h1_full` so với `h1_image_anchors` giữ text cho hard loss/grounding. Đây là ablation **anchor fusion**, không phải loại toàn bộ text khỏi hệ thống.
- Mọi variants hiện vẫn qua caption/concept preparation và `build_anchors()` có negative filter. Vì vậy ngay cả control không dùng negative loss cũng có thể không chạy được trên mẫu bị lỗi bank. Cần báo coverage và dùng cùng tập mẫu khi so ASR; đây là hạn chế của preprocessing dùng chung, không phải bằng chứng baseline cần negatives.

### 10. Bật/tắt loss và chọn text/image encoders

`LOSS_WEIGHTS` điều khiển đóng góp gradient; flags trong `VARIANTS` điều khiển có tính objective trong vòng tối ưu hay không. Đặt weight 0 không tự bỏ forward/clustering tương ứng nếu flag vẫn bật.

Ví dụ thêm vào **cuối Cell 1** để giữ joint/hard objectives cùng dynamic grounding, tắt FOA loss:

```python
RUN_VARIANTS = ['h1_full']
VARIANTS['h1_full']['foa'] = False
VARIANTS['h1_full']['joint'] = True
VARIANTS['h1_full']['hard_loss'] = True
LOSS_WEIGHTS = {'foa': 0.0, 'joint': 1.0, 'hard': 0.5}
```

Tắt joint loss nhưng giữ FOA/hard:

```python
VARIANTS['h1_full']['foa'] = True
VARIANTS['h1_full']['joint'] = False
VARIANTS['h1_full']['hard_loss'] = True
LOSS_WEIGHTS = {'foa': 1.0, 'joint': 0.0, 'hard': 0.5}
```

Tắt riêng hard loss, vẫn giữ mining và negative-conditioned maps:

```python
VARIANTS['h1_full']['hard_loss'] = False
LOSS_WEIGHTS['hard'] = 0.0
```

Tắt cả hard loss và active negative maps:

```python
VARIANTS['h1_full']['negatives'] = 'none'
VARIANTS['h1_full']['hard_loss'] = False
LOSS_WEIGHTS['hard'] = 0.0
```

Cấu hình cuối **vẫn không loại bank khỏi stationary margin của memory hoặc khỏi preprocessing**, như đã giải thích. Muốn thí nghiệm hoàn toàn không có negative bank thì cần thay code preparation/anchors/grounder; chưa được thực hiện chỉ bằng flag này.

Để dùng một CLIP với cả image tower và text tower tương ứng:

```python
BACKBONES = ['B16']
GROUNDING_BACKBONE = 'B16'
```

Không có flag riêng tắt text tower cho toàn pipeline. Tắt `joint` không tắt text, vì hard loss/grounding vẫn cần text embeddings. `fusion=False` hoặc `FUSION_LAMBDA=1.0` làm anchors image-only, nhưng không loại text khỏi hard loss/grounding.

`FOA_LOCAL_WEIGHT=0.0` chỉ loại đóng góp local OT vào FOA loss; code hiện vẫn tính local distance nếu `foa=True`. `build_anchors()` vẫn chuẩn bị target centers và joint anchors kể cả khi objective tương ứng bị tắt. Không đặt cả ba loss weights bằng 0 nếu muốn có attack updates. Mỗi cấu hình thay đổi nên chạy run mới; resume fingerprint không cho ghép với run có config khác.

### 11. Các tham số quan trọng đang có trong notebook

| Tham số                                         | Giá trị tại lần đối chiếu | Ý nghĩa                                                                      |
| ----------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------- |
| `BACKBONES`                                     | `['B16','B32','Laion']`   | CLIP ensemble cho loss/ranking; Laion ViT-G lớn, một B16 là cấu hình nhẹ hơn |
| `GROUNDING_BACKBONE`                            | `B16`                     | Một CLIP cho filtering và relevance/memory margin                            |
| `MAX_SOURCE_CONCEPTS`                           | 3                         | Số source phrases tối đa được giữ                                            |
| `MIN_NEGATIVES`, `MAX_NEGATIVES`                | 2, 6                      | Minimum candidates/bank sau kiểm tra và maximum candidates giữ               |
| `TEXT_RETRIES`                                  | 3                         | Số lần thử concept JSON; không phải số caption retries                       |
| `NEGATIVE_TEXT_SIM_MAX`                         | 0.98                      | Ngưỡng near-duplicate trong CLIP text space                                  |
| `NEGATIVE_TARGET_MARGIN`                        | 0.0                       | Margin lọc negative theo clean target image                                  |
| `HARD_TOP_K`, `MINE_EVERY`                      | 2, 10                     | Số active negatives và chu kỳ chọn lại                                       |
| `REFRESH_EVERY`, `GROUNDING_LAST_LAYERS`        | 30, 6                     | Chu kỳ grounding và số attention layers cuối                                 |
| `CROP_AREA`                                     | 0.5                       | Diện tích window mong muốn trước làm tròn cạnh                               |
| `MEMORY_EMA`, `CREDIT_DECAY`, `CREDIT_STRENGTH` | 0.7, 0.9, 1.0             | Smoothing, decay và mức discount priority                                    |
| `FUSION_LAMBDA`                                 | 0.6                       | Tỷ trọng image trong source/target joint anchors                             |
| `TEMPERATURE`, `POSITIVE_OMEGA`                 | 0.2, 2.0                  | Temperature dùng trong CE/map weights; omega chỉ cho joint positive          |
| `FOA_LOCAL_WEIGHT`                              | 0.2                       | Tỷ trọng local OT trong FOA-derived term                                     |
| `LOSS_WEIGHTS`                                  | 1.0 / 1.0 / 0.5           | FOA / joint / hard                                                           |
| `STEPS`, `EPSILON_255`, `ALPHA_255`             | 300, 16, 1                | Số updates; L∞ budget và step size theo thang 0–255                          |
| `MOMENTUM`                                      | 0.0                       | Projected sign descent mặc định; có thể ablate momentum                      |
| `RUN_ATTACK`, `NUM_SAMPLES`                     | False, 1                  | Mặc định preflight và một cặp, chưa tự chạy full experiment                  |

Những ngưỡng/weights kết hợp ở đây là cấu hình thí nghiệm, không phải bằng chứng chúng tối ưu hoặc làm tăng ASR. Cần tune trên development set, tách khỏi test victims/data dùng để báo cáo transfer.

### 12. Artifacts, resume, lỗi và đo ASR

| Artifact                                | Nội dung / dùng để kiểm tra                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `run_config.json`                       | Config, selected variants, resolved model revisions, data/code hashes và fingerprint             |
| `events.jsonl`                          | Tiến độ, captions, retries, errors, refresh và sample completion                                 |
| `text_records.jsonl`                    | Captions, source concepts, proposed edits/candidates và trạng thái mỗi mẫu                       |
| `samples/<ID>/negative_candidates.json` | Raw concept generation hợp lệ, rejected edits, candidates và CLIP filtering audit                |
| `samples/<ID>/concept_token_spans.json` | Tokens/character offsets của source concepts nếu tokenizer hỗ trợ                                |
| `samples/<ID>/steps.jsonl`              | Loss terms từng CLIP, ensemble weights, active negative IDs/scores, box, L∞ và support fractions |
| `maps/concept_maps_<step>.npy`          | Map riêng: source concepts trước, active negatives sau                                           |
| `maps/relevance_<step>.npy`             | Priority map dùng chọn box; có memory thì không đồng nhất fresh map                              |
| `maps/relevance_<step>.json`            | Refresh step, box, concept margins/weights, stationary margin, progress và số fallback           |
| `maps/image_<step>.png`                 | Snapshot ảnh tại lần refresh để overlay/quan sát                                                 |
| `status.json`, `record.json`            | Trạng thái và kết quả từng mẫu; compute counts, time, peak CUDA memory, file hashes              |
| `images/`, `manifest.jsonl`             | Source/target/adv lossless và input cho evaluation                                               |
| `summary.json`, `failures.json`, ZIP    | Coverage, mẫu lỗi/chưa hoàn tất và gói kết quả                                                   |

IDs negatives trong trace đánh chỉ số vào **bank sau filter**. `candidate_indices` trong audit ánh xạ các vị trí này về original candidate list. `concept_order` cần đọc cùng `source_concept_count`: phần đầu là source-concept indices, phần sau là negative-bank indices, không phải một ID space chung.

Resume chỉ ở ranh giới mẫu: caption record hợp lệ được reuse khi image hashes khớp; mẫu hoàn tất được skip khi file hashes hợp lệ. Mẫu đang dở chạy lại từ source và seed của mẫu, không khôi phục optimizer ở giữa step. `steps.jsonl` được reset cho invocation mới; events giữ lịch sử. Đổi config/code/model/data không được coi là cùng run theo fingerprint.

Attack outputs để `ASR=None`. Muốn đo transfer, dùng cùng victim captioning protocol cho source/target/adv và cùng judge/rubric/threshold cho mọi variants. Nếu dùng quy tắc targeted evaluator theo paper-style trong repo thì success là score > 0.5; không trộn với cấu hình legacy score >= 0.3. Phải ghi rõ protocol thực sự dùng; chỉ đổi threshold không tự tái tạo toàn bộ paper protocol.

Báo cáo ASR kèm denominator, coverage, lỗi, AvgSim, L∞/L2, runtime và compute. Không âm thầm bỏ mẫu lỗi rồi so percentages trên những tập khác nhau. `compute_counts` của optimizer không bao gồm caption generation/build anchors; peak CUDA memory bao gồm model resident. Điều kiện dynamic cần thêm equal-compute/equal-time comparison ngoài equal-step.

### 13. Điểm lấy từ paper và điểm đề xuất trong notebook

| Nguồn                                                                                                        | Điều dùng làm nền / đối chiếu                                                      | Phần không được tự gán cho tác giả                                                                                       |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [FOA-Attack](https://arxiv.org/abs/2505.21494)                                                               | Global/local alignment, patch clustering/OT, ensemble weighting                    | Tổ hợp loss/grounding/memory này; các điều chỉnh K-means, detached OT và cached targets không phải tuyên bố reproduction |
| [MPCAttack](https://arxiv.org/abs/2603.04846)                                                                | Caption encoding, image/text fusion, target/source contrastive objective           | CLIP-only hybrid, bank single-span edits và concept-margin memory không được mô tả như toàn bộ MPC gốc                   |
| [Chefer et al.](https://arxiv.org/abs/2103.15679)                                                            | Gradient-weighted relevance propagation làm cơ sở implementation rollout           | Region scheduler, hard-negative conditioning và credit heuristic là lựa chọn của notebook                                |
| [Grad-ECLIP](https://proceedings.mlr.press/v235/zhao24p.html), [Finer-CAM](https://arxiv.org/abs/2501.11309) | Related XAI và ý tưởng giải thích discriminative evidence                          | Chưa implement hai phương pháp này thành comparators riêng                                                               |
| [CF-VLM](https://arxiv.org/abs/2506.17267), [SADCA](https://arxiv.org/abs/2603.04839v3)                      | Counterfactual/contrastive và dynamic interactions là tiền lệ cần kiểm tra overlap | Không gọi bank hiện tại là phát minh contrastive/counterfactual hoặc bằng chứng causal reasoning                         |
| [GDA](https://arxiv.org/abs/2602.09431v2), [SAGA](https://arxiv.org/abs/2602.04356v2)                        | Grounding-aware allocation và fixed-map region sequencing đã có tiền lệ            | Re-grounding + EMA/credit là cơ chế đề xuất cần kiểm chứng, chưa xác nhận novelty                                        |

Các mô tả phương pháp trong phần đặc tả trên được đối chiếu trực tiếp với code local, không suy ra chi tiết thuật toán chỉ từ abstract. Những abstract/paper links dùng để kiểm tra tiền lệ và nguồn ý tưởng; hiệu quả của tổ hợp phải được chứng minh bằng thí nghiệm mới.

### 14. Ánh xạ tài liệu với code để tự kiểm tra

| Cell trong notebook | Hàm/chức năng                                                                            | Phần đặc tả tương ứng                             |
| ------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Cell 1              | Config, `VARIANTS`, assertions                                                           | Ablations, bật/tắt và tham số                     |
| Cell 3–4            | Trace/hash helpers, `prepare_pairs()`, preprocess                                        | Dữ liệu, output và reproducibility                |
| Cell 5              | `MPCCaptioner.generate()`, `.concepts()`, `parse_concepts()`                             | Caption generation và raw negative candidates     |
| Cell 6              | `ClipBackbone`, `build_anchors()`, `cluster_patches()`, `ot_similarity()`, `anchor_ce()` | Encoders, negative filter, anchors và objectives  |
| Cell 7              | `mine_negatives()`, `ConceptGrounder`, `choose_box()`, `crop_view()`, `support_mask()`   | H1 mining, H2 grounding, H3 memory/regions        |
| Cell 8              | `objective_terms()`, `run_sample()`, `project_image()`                                   | Full/crop loss, ensemble gradient và pixel update |
| Cell 9              | Self-checks và random tiny CLIP check                                                    | Kiểm tra API/gradient; không phải pretrained ASR  |
| Cell 10–12          | Run identity, caption cache, load models, sample/variant loops                           | Toàn bộ run và error/resume branches              |
| Cell 13–14          | Manifest/summary/ZIP và preview                                                          | Export/evaluation handoff và quan sát maps        |

Tài liệu này không thay đổi code attack. Nếu muốn thêm một negative generator, semantic verifier hoặc ablation loại bank hoàn toàn, cần triển khai và kiểm tra riêng; chúng chưa tồn tại chỉ vì được nêu như hạn chế hay hướng kiểm chứng.
