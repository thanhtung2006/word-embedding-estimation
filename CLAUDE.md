# CLAUDE.md — LAB1-claude: Word Embedding Estimation (WE.bin)

## 1. Bài toán
- Tạo word embeddings cho **655 từ tiếng Anh** trong `words/starting-kit.py` (biến `Words`).
- Nộp **file zip chứa `WE.bin`** (word2vec **binary** format, đọc được bằng gensim `KeyedVectors.load_word2vec_format(..., binary=True)`).
- Hệ thống chấm ẩn, nhiều khả năng là **tương đồng từ kiểu SimLex** (Spearman giữa cosine và điểm người gán). Điểm baseline của starting-kit: **0.37**.
- Leaderboard có ≥6 bài đạt ≥0.8, một số ~0.9.
- Không hỏi được ban tổ chức. Nếu có điều chưa rõ, chọn cách hiểu hợp lý nhất và ghi lại.

## 2. RÀNG BUỘC CỨNG (không được vi phạm)
- **Cấm dùng WordSim-353 và SimLex-999 ở bất kỳ bước nào**: không tune, không chọn tham số, không đánh giá, không làm few-shot, không dùng vector được tune trên chúng (ví dụ Paragram-SL999). Vi phạm → bị đưa về điểm baseline.
- Dev set được phép: **RG-65, MC-30, MEN, MTurk-771** (từ `mfaruqui/eval-word-vectors`). Nhưng chúng thiên về *relatedness*, đã bão hòa (~0.9) và sai số bootstrap khoảng ±0.03, nên **không dùng dev để chọn giữa các biến thể gần nhau**. Trước đây dev đã gây hiểu lầm.
- `WE.bin` phải chứa đúng 655 từ trong `Words`, cùng số chiều, float32.

## 3. Cấu trúc thư mục
```
LAB1-claude/
├── CLAUDE.md
├── words/
│   └── starting-kit.py             # danh sách 655 từ (Words) + code baseline
├── code-base/
│   └── WE_pairwise_qwen_v2.ipynb   # notebook hiện tại (chạy trên Kaggle, cần GPU)
├── inputs/
│   └── WE_ensemble6_B2_F2_L2_strong_C_E.zip   # ensemble cổ điển (0.73), input bắt buộc của notebook
├── cache/                          # điểm Qwen đã chấm (tải từ Kaggle /kaggle/working/cache)
│   ├── qwen_main_<key>.npy/.json   # 2 cặp file: prompt 'sim', thứ tự A-B và B-A, mỗi file 214,185 cặp
│   └── qwen_dev_<key>.npy/.json    # 12 cặp file: điểm dev (RG-65, MC-30) cho bảng so sánh cấu hình
└── submissions/                    # các file nộp (zip chứa WE.bin), điểm ở mục 4
    ├── WE_Base_Ensemble_655d.zip
    ├── WE_Qwen_only.zip
    ├── WE_Combined_alpha0.5.zip
    ├── WE_Combined_alpha1.zip
    └── WE_Combined_alpha2.zip
```
- `cache/` là kết quả của cả lần chạy Qwen (~214k cặp × 2 thứ tự). **Không xóa, không ghi đè.** Có nó thì đổi α, đổi cách kết hợp hoặc thêm prompt đều rẻ.
- Cache key = hash(model, 4-bit, dtype, prompt, few-shot, loại, danh sách cặp). Đổi bất kỳ yếu tố nào thì cache cũ không còn khớp và sẽ chấm lại, không bị dùng nhầm.
- Tái sử dụng trên Kaggle: upload `cache/` thành Dataset rồi đặt `INPUT_CACHE_DIR='/kaggle/input/<ten-dataset>/cache'` trong ô config; notebook chép về và chỉ chấm các cặp còn thiếu.
- Muốn thử α hoặc cách kết hợp khác mà không cần GPU: nạp hai file `qwen_main_*.npy` (trung bình hai thứ tự → z-score), kết hợp với ma trận cosine của ensemble trong `inputs/`, rồi dựng vector như mục 5.2 và xuất zip. Kiểm tra file `.json` để biết file nào là thứ tự nào (`kind`, `prompt`, `done`, `n`).

## 4. Lịch sử kết quả (điểm do hệ thống chấm trả về)
| Phương pháp | Điểm |
|---|---|
| Baseline starting-kit | 0.37 |
| Ensemble embedding (Numberbatch×2 + fastText) | 0.59 |
| Counter-fitting mild / medium / strong | 0.60 / 0.68 / 0.69 |
| A (medium, no ANTX) | 0.64 |
| **B** (Numberbatch-only, strong CF) | **0.71** |
| C (strong, rel-repel) | 0.69 |
| D (L1 / L2 R2) | 0.68 / 0.69 |
| Ensemble4 (B×2, L2, strong, C) | 0.72 |
| Sense mix50 / sense-only debiased | 0.72 / 0.70 |
| E (Numberbatch 19.08 + PPDB) / F (L-all top5) | 0.69 / 0.72 |
| **Ensemble6** (B×2, F×2, L2, strong, C, E) | **0.73** |
| PPDB-variants ensemble | 0.73 |
| **Qwen only** (notebook hiện tại) | **0.80** |
| **Combined α=0.5** | **0.83** |
| **Combined α=1** | **0.85** |
| **Combined α=2** | **0.85** |

Nhận xét:
- Các phương pháp cổ điển (embedding pretrained + counter-fitting WordNet/PPDB) **chạm trần ~0.73**. Chênh lệch giữa các biến thể gần nhau (±0.02–0.03) là nhiễu.
- Chuyển sang **LLM chấm điểm từng cặp từ** là bước nhảy lớn: Qwen đứng một mình (0.80) đã hơn ensemble tốt nhất (0.73); kết hợp thêm ensemble lên 0.85.
- α=1 và α=2 bằng nhau (0.85), α=0.5 thấp hơn (0.83): trong vùng nhiễu nên đừng diễn giải quá mức.

## 5. Phương pháp của notebook hiện tại (`WE_pairwise_qwen_v2.ipynb`)
Ý tưởng: embedding tĩnh không nhất thiết phải học từ ngữ liệu. Có thể **dựng vector sao cho cosine = điểm tương đồng** do LLM chấm cho mọi cặp trong 655 từ.

1. **Chấm cặp (655·654/2 ≈ 214k cặp)** bằng `Qwen/Qwen2.5-7B-Instruct`, 4-bit (bitsandbytes). Prompt hỏi mức tương đồng từ 0–9; lấy softmax trên logit của 10 token chữ số (lm_head chỉ giữ 10 id đó), `score = Σ p_d·d / 9`. Chấm **cả hai thứ tự** (A,B) và (B,A) rồi lấy trung bình. Bộ prompt có `sim`, `subst`, `kind`; few-shot (lấy từ RG-65, bỏ cặp có cả hai từ thuộc 655) là tùy chọn.
2. **Điểm → vector**: z-score điểm cặp → ma trận đối xứng S (đường chéo 0) → cộng `c = −λmin + 1e-3` vào đường chéo để nửa xác định dương → `eigh` → `X = V·√Λ`, giữ 655 chiều. Cosine giữa hai từ = S_ij/c, nên **giữ nguyên thứ hạng** điểm. Vì chấm theo thứ hạng nên cosine tuyệt đối gần 0 là chấp nhận được.
3. **Combined**: `z(S_ens) + α·z(S_qwen)`, với `S_ens` là ma trận cosine của ensemble 0.73 trên 655 từ, α ∈ {0.5, 1, 2}.
4. **Cache + checkpoint** theo (model, prompt, danh sách cặp); `STOP_AFTER_HOURS=10.5`; cờ `SMOKE_TEST` (mặc định True) chạy thử 512 cặp để ước lượng tốc độ. Các file Qwen chỉ xuất khi `SMOKE_TEST=False` và chạy xong đủ.

Cấu hình mặc định của notebook: `MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'`, `LOAD_4BIT=True`, `PROMPT_SET=['sim']`, `BOTH_ORDERS=True`, `USE_FEWSHOT=False`, `ALPHA_LIST=[0.5,1,2]`, `OUT_DIM=None`.
Lần chạy cho ra 0.80/0.83/0.85/0.85 dùng **đúng cấu hình mặc định** ở trên: cache cho thấy 2 thành phần `qwen_main` đầy đủ (1 prompt `sim` × 2 thứ tự, mỗi thành phần 214,185 cặp) và 12 file `qwen_dev` (3 prompt × có/không few-shot × 2 thứ tự, chỉ dùng để so sánh trên dev).

Output đã nộp (đều 1.6 MB, nằm trong `submissions/`): `WE_Base_Ensemble_655d.zip` (**chưa có điểm, chưa kiểm chứng**), `WE_Qwen_only.zip`, `WE_Combined_alpha0.5.zip`, `WE_Combined_alpha1.zip`, `WE_Combined_alpha2.zip`.

## 6. Lưu ý / rủi ro
- **Nhiễm benchmark**: Qwen có thể đã thấy SimLex/WordSim lúc huấn luyện. Chúng ta không dùng nhãn của chúng, nhưng điểm cao có thể một phần đến từ chỗ đó. Không tự đánh giá điểm này là "thực lực" mà không nói rõ.
- Không có Qwen thật trong môi trường sandbox của Claude (không GPU, không truy cập Hugging Face). Mọi thử nghiệm Qwen thật chạy trên Kaggle; trong sandbox chỉ test được bằng bộ chấm giả (stub).
- Không chọn α hay prompt theo dev (nhiễu quá lớn). Chỉ dựa vào điểm nộp thật.

## 7. Hướng cải thiện tiếp theo (thứ tự gợi ý)
1. **Ensemble nhiều prompt** (`PROMPT_SET=['sim','subst','kind']`): thường ổn định hơn vì giảm nhiễu từ cách đặt câu hỏi. Dùng lại `cache/` (xem mục 3) nên chỉ phải chấm các thành phần của prompt mới, 2 thành phần `sim` đã có.
2. **Few-shot từ RG-65** (`USE_FEWSHOT=True`, `N_SHOTS=8`): hiệu chỉnh thang điểm cho Qwen. Phải kiểm tra không có cặp nào có cả hai từ thuộc 655 từ (notebook đã lọc).
3. **Mô hình lớn hơn** nếu còn VRAM/thời gian (ví dụ 14B 4-bit), hoặc **ensemble nhiều LLM** khác họ (Qwen + Llama/Gemma), vì lỗi của chúng ít tương quan.
4. Thử α trung gian (1.5) hoặc trọng số theo từng thành phần; chỉ đáng làm nếu thật sự còn lượt nộp.
5. Vẫn bám đúng quy tắc: mỗi lần đổi chỉ đổi một yếu tố để biết nguyên nhân; ghi điểm vào bảng ở mục 4.

## 8. Quy ước khi làm việc ở repo này
- Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc; thuật ngữ kỹ thuật giữ tiếng Anh.
- Trước khi giao file nộp, luôn kiểm tra: zip có đúng 1 file `WE.bin`, đọc được bằng gensim (binary), đủ 655 từ, đồng số chiều, không NaN.
- Không bao giờ tải, đọc hay dùng WordSim-353 / SimLex-999, kể cả "chỉ để xem thử".
- Khi sửa notebook: giữ nguyên cache key, checkpoint và cơ chế `SMOKE_TEST`; chạy thử bằng stub trước khi giao.
