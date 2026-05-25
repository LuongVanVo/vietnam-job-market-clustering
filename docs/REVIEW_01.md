# Review: `01_preprocess.ipynb`
**Dự án:** Phân cụm tin tuyển dụng Việt Nam 
**Phương pháp vector hóa:** TF-IDF + TruncatedSVD 
**Dataset:** `tinixai/vietnamese-job-descriptions`

---

# # 1. Tổng quan

Notebook triển khai đúng mục tiêu giai đoạn tiền xử lý: tải dữ liệu, loại bỏ trường nhiễu, làm sạch văn bản, trích xuất lương/kinh nghiệm, xử lý outlier và lưu file sạch. Cấu trúc rõ ràng, có comment tiếng Việt, dễ theo dõi. Một số điểm cần bổ sung hoặc điều chỉnh để đảm bảo chất lượng cho bước feature engineering tiếp theo.

---

# # 2. Điểm làm tốt

**Tránh data leakage đúng cách**
Tập Train được chia trước khi tính bất kỳ thống kê nào (mode, median, IQR). Các hằng số này sau đó được áp dụng nhất quán cho cả Train lẫn Test. Đây là điểm quan trọng nhất của notebook và được thực hiện chính xác.

**`parse_salary_string` xử lý đa dạng đơn vị**
Hàm xử lý được các định dạng phổ biến trong dữ liệu tuyển dụng VN: triệu VND, nghìn VND, USD, lương tuần. Loại bỏ các chuỗi mơ hồ (`thỏa thuận`, `cạnh tranh`) và có ngưỡng bảo vệ `> 500 triệu` để lọc giá trị bất hợp lý.

**IQR dùng hệ số bất đối xứng**
`q75 + 3.0 * IQR` cho phía trên (thay vì 1.5 chuẩn) là lựa chọn đúng, tránh loại nhầm mức lương C-level hoặc expat hợp lệ.

**Ngưỡng word count hợp lý**
Lọc `< 50` và `> 5000` từ là ngưỡng phù hợp với đặc thù job description tiếng Việt.

**`lowercase` trong `clean_text`**
Hoàn toàn đúng khi dùng TF-IDF. Lowercase giúp gộp các biến thể hoa/thường về cùng token, giảm kích thước vocabulary không cần thiết.

**Trực quan hóa đủ 3 góc**
So sánh word count, phân bố lương và tỷ lệ missing values trước/sau xử lý đủ để báo cáo và kiểm tra nhanh kết quả.

---

# # 3. Vấn đề cần sửa

# # # 3.1 Thiếu Tokenize tiếng Việt — Nghiêm trọng

Đây là vấn đề quan trọng nhất khi dùng TF-IDF với tiếng Việt. TF-IDF mặc định tách token theo khoảng trắng, trong khi tiếng Việt là ngôn ngữ đa âm tiết — một từ có thể gồm nhiều âm tiết cách nhau bằng dấu cách.

```
# Không tokenize → TF-IDF tách sai
"kỹ thuật phần mềm" → ["kỹ", "thuật", "phần", "mềm"] 

# Có tokenize → đúng
"kỹ thuật phần mềm" → ["kỹ_thuật", "phần_mềm"] 
```

Hệ quả: các từ ghép quan trọng như `kỹ_thuật`, `nhân_viên`, `kinh_doanh`, `phần_mềm` bị tách vụn → vector TF-IDF mất đi đặc trưng ngữ nghĩa cốt lõi → chất lượng phân cụm giảm đáng kể.

**Sửa:** Thêm bước tokenize bằng `underthesea` vào cuối hàm `clean_text` hoặc tạo hàm riêng áp dụng sau khi clean:

```python
from underthesea import word_tokenize

def tokenize_vi(text):
 """Tokenize tiếng Việt, trả về chuỗi với từ ghép nối bằng dấu gạch dưới"""
 return word_tokenize(text, format='text')

# Áp dụng sau bước clean_text
df_clean_train['text_tokenized'] = df_clean_train['text_combined'].apply(tokenize_vi)
df_clean_test['text_tokenized'] = df_clean_test['text_combined'].apply(tokenize_vi)

# Dùng text_tokenized (không phải text_combined) làm đầu vào TF-IDF
```

---

# # # 3.2 Thiếu Stopwords tiếng Việt — Quan trọng

Notebook chưa loại bỏ stopwords. Với TF-IDF, các từ xuất hiện gần như trong mọi tin tuyển dụng (như `công ty`, `ứng viên`, `yêu cầu`, `làm việc`) có IDF ≈ 0 → trọng số gần bằng 0, nhưng vẫn chiếm slot trong `max_features` và tạo noise trong vector.

**Sửa:** Xây dựng stopwords list gồm 2 tầng và truyền vào `TfidfVectorizer`:

```python
# Tầng 1: Stopwords tiếng Việt chung
# Nguồn: github.com/stopwords-iso/stopwords-vi
VI_GENERAL_STOPWORDS = {...} # load từ file

# Tầng 2: Stopwords domain tuyển dụng (tự định nghĩa)
VI_RECRUITMENT_STOPWORDS = {
 "công_ty", "ứng_viên", "tuyển_dụng", "vị_trí",
 "yêu_cầu", "mô_tả", "công_việc", "nhân_viên",
 "làm_việc", "thông_tin", "liên_hệ", "nộp_hồ_sơ",
 "chúng_tôi", "bạn", "các", "được", "có_thể",
 "tham_gia", "môi_trường", "cơ_hội", "phát_triển"
}

ALL_STOPWORDS = VI_GENERAL_STOPWORDS | VI_RECRUITMENT_STOPWORDS

# Truyền vào vectorizer
tfidf = TfidfVectorizer(
 stop_words=list(ALL_STOPWORDS),
 ...
)
```

---

# # # 3.3 Thiếu bước gộp Rare Categories — Quan trọng

Notebook drop `id`, `company_name`, `year` nhưng không xử lý các giá trị hiếm trong `job_industry` và `location`. Khi One-Hot Encoding ở bước sau, các category chỉ xuất hiện 1–2 lần sẽ tạo ra chiều vector gần toàn số 0 → noise.

**Sửa:** Thêm vào cuối bước xử lý Train, trước khi lưu file:

```python
def merge_rare_categories(series_train, series_test, threshold=0.001):
 """
 Gộp categories hiếm vào nhãn 'Khác'.
 Tính ngưỡng từ train, áp dụng cho cả train và test.
 """
 freq = series_train.value_counts(normalize=True)
 rare = set(freq[freq < threshold].index)
 return (
 series_train.apply(lambda x: 'Khác' if x in rare else x),
 series_test.apply(lambda x: 'Khác' if x in rare else x)
 )

for col in ['job_industry', 'location']:
 df_clean_train[col], df_clean_test[col] = merge_rare_categories(
 df_clean_train[col], df_clean_test[col], threshold=0.001
 )
```

---

# # # 3.4 Salary floor bằng 0 — Trung bình

```python
sal_cap_lower = max(0.0, q25 - 1.5 * iqr)
```

Nếu Q1 nhỏ, công thức trên trả về `0.0`. Khi dùng `.clip(lower=0.0)`, các tin có lương nhập sai = 0 không bị lọc. Sau khi fillna bằng median, giá trị 0 này vẫn tồn tại và kéo lệch phân phối.

**Sửa:** Đặt floor thực tế:

```python
SALARY_MIN_FLOOR = 1.0 # 1 triệu VND — mức tối thiểu hợp lý
sal_cap_lower = max(SALARY_MIN_FLOOR, q25 - 1.5 * iqr)
```

---

# # # 3.5 Regex số điện thoại chưa phủ hết pattern — Nhỏ

```python
# Hiện tại — bỏ sót dạng có dấu chấm/gạch ngang
text = re.sub(r'(?:\+84|0)(?:\s*\d){9,10}', ' ', text)
```

Bỏ sót các định dạng phổ biến: `028.3456.7890`, `(028) 3456 7890`, `0912-345-678`.

**Sửa:**

```python
text = re.sub(r'(\+84|0)[\s\.\-]?(\d[\s\.\-]?){8,10}\d', ' ', text)
```

---

# # # 3.6 `experience_level` đóng 2 vai chưa được ghi chú rõ — Nhỏ

Cột `experience_level` vừa được dùng làm **categorical feature** (nằm trong `cat_cols`, fillna bằng mode) vừa được **parse sang số** (`exp_min_years`, `exp_max_years`). Logic này đúng nhưng không có comment giải thích → dễ gây nhầm lẫn ở notebook sau.

**Sửa:** Thêm comment tại điểm khai báo:

```python
cat_cols = [
 'location', 'job_type', 'job_industry',
 'experience_level', # Giữ để OHE; đồng thời parse sang exp_min/max_years bên dưới
 'education_level', 'job_position'
]
```

---

# # 4. Ghi chú cho bước tiếp theo (notebook 02)

Notebook 02 cần thực hiện thêm các bước sau (không thuộc phạm vi notebook này nhưng cần biết để chuẩn bị đúng output):

| Bước | Nội dung | Lý do |
|---|---|---|
| TF-IDF | `fit` trên train, `transform` cả train & test | Tránh leakage |
| TruncatedSVD | Giảm từ ~10,000 → 300 chiều | Curse of dimensionality |
| Kiểm tra variance | `svd.explained_variance_ratio_.sum()` | Kỳ vọng 40–60% |
| OHE structured fields | `job_industry`, `location`, `job_type` | Fit trên train only |
| Normalize salary | `MinMaxScaler` fit trên train | Fit trên train only |
| Concat feature vector | Text (300D) + Structured (~60D) | Ra ~360 chiều |
| UMAP | Giảm về 20D để cluster, 2D để visualize | Dùng `cosine` metric |

---

# # 5. Tóm tắt

| Hạng mục | Trạng thái | Mức độ |
|---|---|---|
| Cấu trúc pipeline tổng thể | Tốt | — |
| Tránh data leakage | Đúng | — |
| `lowercase` cho TF-IDF | Đúng | — |
| Parse salary đa đơn vị | Đầy đủ | — |
| IQR hệ số bất đối xứng | Đúng | — |
| Ngưỡng word count | Hợp lý | — |
| **Tokenize tiếng Việt** | Thiếu | 🔴 Cao |
| **Stopwords tiếng Việt** | Thiếu | 🔴 Cao |
| **Rare categories** | Thiếu | 🟠 Trung bình |
| Salary floor > 0 | Chưa đặt | 🟠 Trung bình |
| Regex SĐT | Chưa phủ hết | 🟡 Nhỏ |
| Comment `experience_level` | Thiếu rõ ràng | 🟡 Nhỏ |

**Thứ tự ưu tiên sửa:** Tokenize tiếng Việt → Stopwords → Rare categories → Salary floor → còn lại.