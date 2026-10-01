# XAI Survey for Caption-Token Grounding and Transferable Black-box VLM Attacks

**Ngày rà soát:** 2026-09-27  
**Mục tiêu:** tìm 5 paper **đã được accept/publish** ở venue lớn hoặc journal uy tín, có **source code công khai**, phục vụ pipeline:

```text
image
  ↓
VLM caption
  ↓
object / noun token
  ↓
XAI: token ↔ image patch / important region
  ↓
kiểm tra token có thật sự được visual evidence hỗ trợ hay không
  ↓
xây importance mask
  ↓
dùng mask để hướng dẫn transferable adversarial attack
  ↓
đánh giá trên black-box VLM
```

## Quy ước đọc survey

Tôi tách rõ hai loại thông tin:

- **[Paper]**: thông tin được paper/abstract hoặc official project/code hỗ trợ trực tiếp.
- **[Research inference]**: suy luận cách ghép paper vào pipeline của bạn. Đây **không phải claim của paper** và cần được kiểm chứng bằng thực nghiệm.

> **Lưu ý:** XAI/attention map không phải ground-truth object detector. Một token có grounding score cao là bằng chứng rằng model đang dùng một vùng ảnh tương ứng, nhưng không tự động chứng minh object đó tồn tại đúng theo nghĩa ground truth. Vì vậy nên xem các score dưới đây là **evidence/confidence signals**.

---

# 1. Danh sách 5 paper

| # | Paper | Venue | Vai trò cho pipeline | Code |
|---|---|---|---|---|
| 1 | **Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations** | CVPR 2026 | Object token → patch/region; phát hiện token không grounded | Có |
| 2 | **Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models** | CVPR 2026 | Visual attention map; tìm question-relevant regions; attention dispersion | Có |
| 3 | **PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models** | CVPR 2026 | Kiểm tra object token có bị language context chi phối thay vì image evidence | Có |
| 4 | **PA-Attack: Guiding Gray-Box Attacks on LVLM Vision Encoders with Prototypes and Attention** | CVPR 2026 | Dùng token-level attention để tập trung perturbation vào critical visual tokens | Có |
| 5 | **Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models via Diffusion Models (AdvDiffVLM)** | IEEE TIFS 2025 | GradCAM-guided mask → transferable attack → commercial black-box VLM | Có |

---

# 2. Paper 1 — Beyond the Global Scores

## Thông tin

**Tuan Dung Nguyen et al.**  
**Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations**  
**CVPR 2026**

### Nguồn kiểm tra

- CVPR 2026 project page:  
  https://token-grounding-detection-cvpr26.github.io/
- CVPR 2026 official program có tên paper:  
  https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf
- Paper/abstract:  
  https://arxiv.org/abs/2604.04863
- Official code được project page trỏ tới:  
  https://github.com/tuandung2812alt3/token-grounding-detector

## Abstract nói gì?

**[Paper]** Paper chỉ ra hạn chế của các detector dựa trên **global image-level relevance**. Một hallucinated token có thể có nhiều tương quan yếu, rải rác trên nhiều local patches; khi cộng lại, global score vẫn cao và detector có thể bị đánh lừa.

**[Paper]** Insight trung tâm:

> Một faithful object token phải được grounded mạnh vào một vùng ảnh cụ thể.

Paper phân tích token-level interactions theo layer và báo cáo hai dấu hiệu của hallucinated token:

1. **Diffuse / non-localized attention** thay vì compact attention vào một vùng cụ thể.
2. **Weak semantic alignment**: token không align mạnh với bất kỳ visual region nào.

## Hai signal quan trọng

### 2.1 Attention Dispersion Score — ADS

**[Paper]** ADS đo mức độ spatially compact của cross-modal attention.

Project page mô tả pipeline:

```text
attention
   ↓
top patches
   ↓
connected components
   ↓
foreground mass + background entropy
   ↓
ADS
```

Interpretation trên project page:

```text
Low ADS  → compact focus   → grounded
High ADS → scattered focus → hallucination-like
```

### 2.2 Cross-modal Grounding Consistency — CGC

**[Paper]** Với token representation \(z_t^{(n)}\) và image-patch representation \(v_p^{(n)}\):

\[
S_{t,p}^{(n)}
=
\frac{
\langle z_t^{(n)},v_p^{(n)}\rangle
}{
\|z_t^{(n)}\|_2\|v_p^{(n)}\|_2
}
\]

Paper/project tạo **per-token patch similarity map** rồi aggregate top-\(k\) patches thành grounding score.

Ý nghĩa:

```text
caption token "dog"
        ↓
similarity with every image patch
        ↓
patch heatmap
        ↓
high-similarity region
```

Đây chính là loại output gần nhất với nhu cầu:

```text
object token → vùng ảnh tương ứng
```

## Dùng vào pipeline của bạn

**[Research inference]**

Ví dụ caption:

```text
"A cyclist wearing a green jersey is riding beside a car."
```

Tách:

```text
cyclist
green jersey
car
```

Sau đó với mỗi token/noun phrase:

```text
token
 ├── ADS → token có focus spatially không?
 └── CGC → patch nào align semantic với token?
```

Bạn có thể giữ region khi:

```text
CGC cao
AND
ADS cho thấy attention đủ localized
```

và loại/bớt trọng số token khi:

```text
CGC thấp
OR
attention quá diffuse
```

### Mức độ phù hợp

**Rất cao** cho bước:

> `caption token → important image patches → token-grounding confidence`

---

# 3. Paper 2 — Deeper Thought, Weaker Aim / VRGA

## Thông tin

**Ruiying Peng et al.**  
**Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models**  
**CVPR 2026**

### Nguồn kiểm tra

- CVPR Open Access:  
  https://openaccess.thecvf.com/content/CVPR2026/html/Peng_Deeper_Thought_Weaker_Aim_Understanding_and_Mitigating_Perceptual_Impairment_during_CVPR_2026_paper.html
- PDF accepted version:  
  https://openaccess.thecvf.com/content/CVPR2026/papers/Peng_Deeper_Thought_Weaker_Aim_Understanding_and_Mitigating_Perceptual_Impairment_during_CVPR_2026_paper.pdf
- Official code:  
  https://github.com/Ivine11/VRGA

## Abstract nói gì?

**[Paper]** Paper nghiên cứu perceptual impairment của MLLM khi reasoning dài. Họ quan sát:

- visual attention bị **scattered**;
- attention **drifts away from question-relevant regions**;
- reasoning prompts có thể làm giảm attention lên vùng quan trọng.

Paper đề xuất **Visual Region-Guided Attention (VRGA)**:

1. chọn visual heads bằng entropy-focus criterion;
2. reweight attention;
3. hướng model quay lại question-relevant visual regions.

Abstract báo cáo cải thiện visual grounding/reasoning accuracy và cung cấp interpretable insights về visual processing.

## Code có gì đáng dùng?

Official repository cho:

- Qwen2.5-VL;
- Qwen3-VL;
- attention intervention;
- evaluation pipeline.

Repo ghi environment khuyến nghị:

```text
Python >= 3.10
CUDA >= 12.1
PyTorch >= 2.4
transformers == 4.52.4
```

và dùng customized Qwen modeling file để lấy/chỉnh attention.

## Dùng vào pipeline của bạn

**[Research inference]**

Paper này hữu ích để lấy **spatial attention prior**:

```text
image + query/object phrase
        ↓
visual attention
        ↓
region-relevance map
```

Ví dụ:

```text
query = "green jersey cyclist"
```

Bạn có thể dùng attention map từ visual heads như một region proposal, sau đó dùng **CGC của Paper 1** để kiểm tra semantic token–patch alignment.

Một cách ghép:

```text
VRGA-style attention region
            ↓
candidate patches
            ↓
CGC(token, patch)
            ↓
verified token-grounded region
```

### Hạn chế

**[Paper-derived scope]** VRGA tập trung vào VQA/reasoning và question-relevant regions; paper không được thiết kế riêng cho adversarial attack.

### Mức độ phù hợp

**Cao** cho:

> `image → salient/question-relevant region`

nhưng **Paper 1 phù hợp hơn** nếu mục tiêu chính là:

> `caption object token → exact supporting patches`.

---

# 4. Paper 3 — PAS

## Thông tin

**Nhat Hoang, Minh Vu, My T. Thai, Manish Bhattarai**  
**PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models**  
**CVPR 2026**

### Nguồn kiểm tra

- CVPR Open Access:  
  https://openaccess.thecvf.com/content/CVPR2026/html/Hoang_PAS_Prelim_Attention_Score_for_Detecting_Object_Hallucinations_in_Large_CVPR_2026_paper.html
- Accepted PDF:  
  https://openaccess.thecvf.com/content/CVPR2026/papers/Hoang_PAS_Prelim_Attention_Score_for_Detecting_Object_Hallucinations_in_Large_CVPR_2026_paper.pdf
- Official code:  
  https://github.com/lanl/PAS

## Abstract nói gì?

**[Paper]** Paper phát hiện rằng trong nhiều hallucinated predictions, LVLM:

```text
giảm phụ thuộc vào image
        +
phụ thuộc nhiều vào previously generated tokens
```

Các token trước đó được paper gọi là **prelim tokens**.

Paper định lượng hiện tượng bằng conditional mutual information và đề xuất:

**Prelim Attention Score (PAS)**

Đặc điểm được abstract nêu:

- training-free;
- dùng attention weights trên prelim tokens;
- không cần additional forward pass;
- tính được ngay trong inference;
- dùng để detect object hallucination.

## Nó có tạo image-region heatmap không?

**Không phải mục tiêu chính.**

PAS chủ yếu trả lời câu hỏi:

```text
object token này được sinh dựa trên visual evidence
hay bị language context trước đó kéo đi?
```

Do đó PAS nên dùng như **reliability signal**, không dùng thay cho patch grounding.

## Dùng vào pipeline của bạn

**[Research inference]**

Giả sử caption có token:

```text
"traffic light"
```

Ta có:

```text
CGC/ADS
    ↓
có region cụ thể trong image support token không?

PAS
    ↓
generation của token có bị previous text chi phối quá mạnh không?
```

Bạn có thể tạo confidence rule:

\[
C_{\text{object}}
=
f(
C_{\text{grounding}},
C_{\text{spatial}},
C_{\text{language-prior}}
)
\]

Trong đó:

- `grounding`: CGC;
- `spatial`: ADS / visual-attention region;
- `language-prior`: PAS.

> Công thức combine cụ thể là **research design**, không phải công thức của PAS.

### Mức độ phù hợp

**Rất cao** cho bước:

> `check captioned object có dấu hiệu hallucination hay không`.

---

# 5. Paper 4 — PA-Attack

## Thông tin

**Hefei Mei, Zirui Wang, Chang Xu, Jianyuan Guo, Minjing Dong**  
**PA-Attack: Guiding Gray-Box Attacks on LVLM Vision Encoders with Prototypes and Attention**  
**CVPR 2026**

### Nguồn kiểm tra

- CVPR official 2026 listing/video page có paper:  
  https://cvpr.thecvf.com/Conferences/2026/Videos
- Paper/abstract:  
  https://arxiv.org/abs/2602.19418
- Author publication page ghi CVPR 2026 acceptance:  
  https://hefeimei06.github.io/
- Official code:  
  https://github.com/hefeimei06/PA-Attack

## Abstract nói gì?

**[Paper]** PA-Attack dùng vision encoder như một **gray-box pivot**.

Hai-stage attention mechanism trong abstract:

1. **token-level attention scores** được dùng để tập trung perturbation vào **critical visual tokens**;
2. attention weights được adaptively recalibrate trong quá trình attack để theo dõi attention thay đổi khi adversarial example được optimize.

Đây là paper quan trọng vì nó chứng minh trực tiếp cách:

```text
visual-token importance
        ↓
perturbation allocation
```

được đưa vào adversarial attack.

## Code có gì?

Official repo chứa:

```text
attention/
prototype/
CLIP_benchmark/
CLIP_eval/
llava/
open_flamingo/
Qwen3-VL/
pope_eval/
vlm_eval/
```

Repo công bố environment chính:

```text
CUDA = 11.8
Python = 3.11
```

và có scripts cho captioning, VQA, POPE, Qwen3-VL, InternVL2.

## Dùng vào pipeline của bạn

**[Research inference]**

Thay vì attack toàn ảnh đồng đều:

\[
g_{\text{adv}} = g
\]

ta có thể nghiên cứu region/token mask:

\[
g_{\text{guided}}
=
M_{\text{XAI}}\odot g
\]

trong đó \(M_{\text{XAI}}\) được tạo từ:

```text
caption object token
        ↓
ADS + CGC
        ↓
validated object patches
        ↓
importance mask
```

Sau đó:

```text
importance mask
        ↓
guide adversarial optimization
```

> Phép nhân mask trên là **research inference**; không nên trích nó như công thức nguyên bản của PA-Attack nếu paper không định nghĩa đúng như vậy.

### Hạn chế rất quan trọng

PA-Attack là **gray-box**, không phải pure closed-source black-box attack.

Vai trò phù hợp nhất trong thesis của bạn là:

```text
surrogate-side XAI / attention-guided attack
        ↓
transfer adversarial image
        ↓
black-box target evaluation
```

---

# 6. Paper 5 — AdvDiffVLM

## Thông tin

**Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo**  
**Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models via Diffusion Models**  
**IEEE Transactions on Information Forensics and Security (TIFS), vol. 20, pp. 1333–1348, 2025**

### Nguồn kiểm tra

- DBLP publication record:  
  https://dblp.org/rec/journals/tifs/GuoPJLG25
- DOI:  
  https://doi.org/10.1109/TIFS.2024.3518072
- Abstract/preprint:  
  https://arxiv.org/abs/2404.10335
- Official code:  
  https://github.com/gq-max/AdvDiffVLM

> DOI mang năm 2024, trong khi journal volume được DBLP ghi là **TIFS 20 (2025), 1333–1348**.

## Abstract nói gì?

**[Paper]** AdvDiffVLM sử dụng diffusion model để tạo natural, unrestricted, targeted adversarial examples.

Hai module được abstract mô tả:

### Adaptive Ensemble Gradient Estimation — AEGE

Sửa score trong reverse diffusion generation để đưa targeted adversarial semantics vào ảnh.

### GradCAM-guided Mask

**[Paper]** Paper sử dụng **GradCAM-guided mask** để điều khiển phân bố spatial của adversarial semantics thay vì để semantics tập trung ở một vùng duy nhất.

**[Paper]** Abstract cũng nêu rằng adversarial examples của AdvDiffVLM có khả năng transfer và có thể attack commercial VLMs trong **black-box environment**, trong đó có GPT-4V.

## Vì sao paper này rất quan trọng cho hướng của bạn?

Đây là precedent trực tiếp:

```text
XAI: GradCAM
    ↓
spatial mask
    ↓
adversarial generation
    ↓
transfer
    ↓
commercial black-box VLM
```

Nó rất gần với hypothesis bạn đang xây:

```text
token-grounded XAI mask
        ↓
guide perturbation
        ↓
improve black-box transferability
```

## Dùng vào pipeline của bạn

**[Research inference]**

Thay vì dùng GradCAM class-region map như AdvDiffVLM, bạn có thể thử:

```text
ADS + CGC token-grounding map
```

hoặc:

```text
visual-attention region + CGC semantic verification
```

để tạo mask cho attack.

Điểm cần kiểm chứng thực nghiệm:

- token-grounding map có tốt hơn GradCAM/random crop không?
- map có làm attack overfit vào surrogate không?
- mask hard hay soft tốt hơn?
- perturb important region hay surrounding/context region tốt hơn cho transfer?

---

# 7. So sánh 5 paper theo đúng bài toán

| Paper | Image region map | Liên hệ generated token ↔ patch | Hallucination/object check | Dùng XAI cho attack | Black-box transfer |
|---|---:|---:|---:|---:|---:|
| Beyond Global Scores | **Có** | **Rất trực tiếp** | **Có** | Chưa trực tiếp | Không phải mục tiêu |
| VRGA | **Có** | Gián tiếp qua attention/query | Gián tiếp | Chưa trực tiếp | Không phải mục tiêu |
| PAS | Không phải output chính | Token-level | **Có** | Không | Không phải mục tiêu |
| PA-Attack | **Có importance trên visual tokens** | Token-level attention | Không phải mục tiêu | **Có** | Transfer/gray-box |
| AdvDiffVLM | **Có, GradCAM mask** | Không phải caption-token grounding | Không | **Có** | **Có** |

---

# 8. Pipeline nên thử trước cho thesis

## Stage 1 — Caption

```text
image
  ↓
open-source VLM
  ↓
caption
```

Ví dụ:

```text
"A man wearing a yellow helmet is riding a bicycle."
```

## Stage 2 — Extract object-bearing tokens

```text
man
yellow helmet
bicycle
```

Bạn có thể làm ở mức:

- noun;
- noun phrase;
- token positions tương ứng trong tokenizer của LVLM.

## Stage 3 — Token → image patch grounding

Paper chính: **Beyond the Global Scores**.

```text
object token
    ↓
ADS
+
CGC(token, each visual patch)
    ↓
token-grounding heatmap
```

Output mong muốn:

```text
"helmet"
   ↓
patch indices / spatial map
   ↓
region around helmet
```

## Stage 4 — Reliability / hallucination filtering

Paper chính: **PAS**.

```text
object token
   ├── visual grounding strong?
   └── language-context dependence suspicious?
```

Ví dụ:

```text
CGC high + ADS compact + PAS reliable
          ↓
keep object region

CGC low + ADS diffuse + PAS suspicious
          ↓
reject / down-weight token
```

> Logic combine này cần calibrate; không phải threshold được các paper trên quy định chung.

## Stage 5 — Build XAI importance map

Một soft mask:

\[
M \in [0,1]^{H\times W}
\]

có thể đến từ:

```text
CGC patch similarity
      ×
spatial attention prior
      ×
token reliability
```

Đây là proposed research design.

## Stage 6 — XAI-guided attack

Thử hai family:

### A. PA-Attack-inspired

```text
important visual tokens
        ↓
concentrate/reweight perturbation
```

### B. AdvDiffVLM-inspired

```text
XAI spatial map
        ↓
attack mask
        ↓
adversarial generation
```

## Stage 7 — Black-box evaluation

XAI chỉ cần chạy trên surrogate/open-source model:

```text
CLIP / Qwen-VL / LLaVA / other surrogate
                ↓
        adversarial image
                ↓
      closed-source target
GPT / Gemini / Claude / other API
```

Không cần truy cập gradient/attention của closed-source target.

---

# 9. Liên hệ với FOA-Attack / random crop

Trong FOA/M-Attack-style optimization, một baseline spatial transformation là:

```text
RandomResizedCrop
```

Research direction của bạn có thể chuyển từ:

```text
random region
```

sang:

```text
semantically validated important region
```

Một ablation hợp lý:

```text
B0: full-image attack
B1: random crop
B2: raw attention-guided region
B3: CGC-guided region
B4: ADS + CGC grounded region
B5: ADS + CGC + PAS filtered region
```

Sau đó đo:

- ASR trên surrogate;
- ASR trên black-box target;
- AvgSim / semantic-target score;
- perturbation concentration;
- transfer gap;
- runtime overhead.

---

# 10. Research question và hypothesis có thể dùng

## Research question

> **Can semantically verified token-grounding maps be used to guide adversarial perturbations and improve transferability to closed-source VLMs compared with random or global image perturbation strategies?**

## Hypothesis

> **Perturbations guided by regions that are both spatially localized and semantically aligned with caption object tokens will transfer better than perturbations guided only by random crops or raw attention.**

Đây là **hypothesis đề xuất**, không phải kết luận của 5 paper.

---

# 11. Paper nào nên chạy trước?

Nếu mục tiêu ngay bây giờ là:

> “Cho ảnh + caption, tôi cần biết object/token nào đang dựa vào vùng nào của ảnh.”

thì thứ tự:

```text
1. Beyond the Global Scores
          ↓
2. PAS
          ↓
3. VRGA
```

Sau khi XAI/token-grounding pipeline chạy ổn:

```text
4. PA-Attack
          ↓
5. AdvDiffVLM
```

Lý do:

- **Beyond Global Scores** gần nhất với `token → patch heatmap`.
- **PAS** giúp kiểm tra token có dấu hiệu bị language prior kéo đi không.
- **VRGA** bổ sung spatial-attention analysis trên Qwen2.5-VL/Qwen3-VL.
- **PA-Attack** cho cách chuyển importance của visual token thành attack guidance.
- **AdvDiffVLM** cho precedent mạnh về XAI/GradCAM mask trong transferable black-box VLM attack.

---

# 12. Một paper rất sát bài toán nhưng tôi không đưa vào 5 paper chính

**Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination — CVPR 2026**

Paper này rất phù hợp về mặt ý tưởng: nó cho thấy **attention magnitude alone không đủ**, và dùng Logit Lens để kiểm tra high-attention region có thật sự decode thành target object semantic hay không.

CVPR source:
https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html

Repository:
https://github.com/wzczc/SADT

**Lý do không đưa vào 5 paper chính:** tại thời điểm rà soát, repository công khai tôi kiểm tra được chủ yếu mới có README, chưa thấy implementation files đủ rõ để tôi khẳng định bạn có thể clone và chạy method end-to-end ngay. Vì constraint của bạn yêu cầu **paper + source code để chạy**, tôi không dùng paper này để chiếm một slot trong 5 paper chính.

---

# 13. Source checklist

## Paper 1 — Beyond the Global Scores
- Venue/project: https://token-grounding-detection-cvpr26.github.io/
- CVPR program: https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf
- Paper: https://arxiv.org/abs/2604.04863
- Code: https://github.com/tuandung2812alt3/token-grounding-detector

## Paper 2 — VRGA
- CVPR: https://openaccess.thecvf.com/content/CVPR2026/html/Peng_Deeper_Thought_Weaker_Aim_Understanding_and_Mitigating_Perceptual_Impairment_during_CVPR_2026_paper.html
- Code: https://github.com/Ivine11/VRGA

## Paper 3 — PAS
- CVPR: https://openaccess.thecvf.com/content/CVPR2026/html/Hoang_PAS_Prelim_Attention_Score_for_Detecting_Object_Hallucinations_in_Large_CVPR_2026_paper.html
- Code: https://github.com/lanl/PAS

## Paper 4 — PA-Attack
- CVPR listing: https://cvpr.thecvf.com/Conferences/2026/Videos
- Paper: https://arxiv.org/abs/2602.19418
- Code: https://github.com/hefeimei06/PA-Attack

## Paper 5 — AdvDiffVLM
- DBLP/TIFS publication: https://dblp.org/rec/journals/tifs/GuoPJLG25
- DOI: https://doi.org/10.1109/TIFS.2024.3518072
- Paper: https://arxiv.org/abs/2404.10335
- Code: https://github.com/gq-max/AdvDiffVLM
