---
name: vn-job-clustering-preprocessing
description: >
 Xử lý dữ liệu (preprocessing pipeline) cho bài toán phân cụm tin tuyển dụng
 tiếng Việt, đặc biệt là dataset tinixai/vietnamese-job-descriptions hoặc
 các dataset tuyển dụng Việt Nam tương tự (scraped từ VietnamWorks, TopCV,
 CareerViet). Bao gồm toàn bộ các bước: làm sạch raw data, xử lý outlier đa
 tầng, encode structured fields, vector hóa text tiếng Việt (PhoBERT/TF-IDF),
 và giảm chiều (UMAP/PCA) trước khi đưa vào clustering. Sử dụng skill này
 bất cứ khi nào người dùng đề cập đến: xử lý dữ liệu tuyển dụng, làm sạch
 job description tiếng Việt, chuẩn bị features cho clustering, hay pipeline
 NLP cho text tuyển dụng Việt Nam.
---

# VN Job Clustering – Preprocessing Pipeline

# # Tổng quan pipeline

```
Raw Data (~18–22 cols)
 │
 ├── [BƯỚC 1] Drop fields không dùng
 ├── [BƯỚC 2] Làm sạch Text (Vietnamese NLP)
 ├── [BƯỚC 3] Xử lý Outlier Tầng 1 – Raw level
 ├── [BƯỚC 4] Encode Structured Fields
 ├── [BƯỚC 5] Vector hóa Text Fields
 ├── [BƯỚC 6] Xử lý Outlier Tầng 2 – Vector space
 ├── [BƯỚC 7] Kết hợp Feature Vector
 └── [BƯỚC 8] Giảm chiều (UMAP/PCA)
 ↓
 Clean Feature Matrix
 (~90–95% mẫu gốc × 10–50 chiều)
```

---

# # BƯỚC 1 – Drop Fields không dùng

# # # Fields cần giữ lại

| Field | Nhóm | Ghi chú |
|---|---|---|
| `title` / `job_title` | Text | Tên vị trí |
| `description` | Text | Mô tả công việc |
| `requirements` | Text | Yêu cầu ứng viên |
| `benefits` | Text | Phúc lợi |
| `industry` / `category` | Structured | Ngành nghề |
| `location` / `city` | Structured | Địa điểm |
| `experience_required` | Structured | Kinh nghiệm |
| `education_level` | Structured | Trình độ học vấn |
| `job_type` | Structured | Loại hình công việc |
| `salary_min` | Structured | Lương tối thiểu |
| `salary_max` | Structured | Lương tối đa |
| `company_size` | Structured | Quy mô công ty |

# # # Fields cần DROP ngay

```python
DROP_FIELDS = [
 'job_id', 'url', 'slug', # Định danh kỹ thuật
 'posted_date', 'expired_date', # Thời gian – không liên quan ngữ nghĩa
 'company_name', # Quá cụ thể, gây noise
 'contact_email', 'phone', # PII
 'view_count', 'apply_count', # Engagement metrics
 'company_description', # Thông tin công ty, không phải JD
]
```

> **Lưu ý:** Sau khi load dataset thực tế, chạy `df.columns.tolist()` để
> đối chiếu tên field chính xác – tên có thể khác với tên ở trên.

---

# # BƯỚC 2 – Làm sạch Text (Vietnamese NLP)

# # # 2.1 Làm sạch cơ bản (áp dụng cho tất cả text fields)

```python
import re

def basic_clean(text: str) -> str:
 if not isinstance(text, str):
 return ""
# Loại bỏ HTML tags còn sót
 text = re.sub(r'<[^>]+>', ' ', text)
# Loại bỏ URL
 text = re.sub(r'http\S+|www\.\S+', ' ', text)
# Loại bỏ email
 text = re.sub(r'\S+@\S+', ' ', text)
# Loại bỏ số điện thoại VN
 text = re.sub(r'(\+84|0)[0-9]{8,10}', ' ', text)
# Chuẩn hóa khoảng trắng
 text = re.sub(r'\s+', ' ', text).strip()
# Loại bỏ ký tự đặc biệt không phải tiếng Việt
 text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
 return text
```

# # # 2.2 Tokenize tiếng Việt

```python
# Ưu tiên 1: underthesea (chính xác hơn)
from underthesea import word_tokenize
tokens = word_tokenize(text, format='text')

# Ưu tiên 2: pyvi (nhẹ hơn, nhanh hơn)
from pyvi import ViTokenizer
tokens = ViTokenizer.tokenize(text)
```

# # # 2.3 Loại bỏ stopwords tiếng Việt

```python
# Dùng stopwords list từ: github.com/stopwords-iso/stopwords-vi
# Kết hợp thêm stopwords domain-specific tuyển dụng:
RECRUITMENT_STOPWORDS = {
 'công_ty', 'ứng_viên', 'tuyển_dụng', 'vị_trí',
 'yêu_cầu', 'mô_tả', 'công_việc', 'nhân_viên',
 'làm_việc', 'thông_tin', 'liên_hệ', 'nộp_hồ_sơ'
}
```

# # # 2.4 Concatenate text fields

```python
# Ghép theo thứ tự ưu tiên thông tin: title > requirements > description > benefits
def concat_text(row):
 parts = []
 for field in ['title', 'requirements', 'description', 'benefits']:
 val = str(row.get(field, '') or '').strip()
 if val:
 parts.append(val)
 return ' '.join(parts)

df['text_combined'] = df.apply(concat_text, axis=1)
```

---

# # BƯỚC 3 – Xử lý Outlier Tầng 1 (Raw Level)

# # # 3.1 Outlier theo độ dài text

```python
# Tính word count sau khi clean
df['word_count'] = df['text_combined'].apply(lambda x: len(x.split()))

# Ngưỡng loại bỏ
MIN_WORDS = 50 # Quá ít thông tin – không embed được tốt
MAX_WORDS = 5000 # Có thể là copy-paste văn bản không liên quan

mask_text = (df['word_count'] >= MIN_WORDS) & (df['word_count'] <= MAX_WORDS)
df = df[mask_text].copy()
# Log: ghi lại số mẫu bị loại
```

**Kỳ vọng loại:** ~2–5% tổng dataset

# # # 3.2 Outlier theo Salary (IQR Rule)

```python
def remove_salary_outliers(df, col_min='salary_min', col_max='salary_max'):
# Chỉ xét các hàng có salary hợp lệ (> 0)
 salary_df = df[(df[col_min] > 0) & (df[col_max] > 0)].copy()

# Tính IQR cho salary_min
 Q1 = salary_df[col_min].quantile(0.25)
 Q3 = salary_df[col_min].quantile(0.75)
 IQR = Q3 - Q1

 lower = Q1 - 1.5 * IQR
 upper = Q3 + 3.0 * IQR # Hệ số 3.0 phía trên vì lương C-level thực sự tồn tại

# Cap thay vì xóa – giữ lại mẫu, chỉ điều chỉnh giá trị
 df[col_min] = df[col_min].clip(lower=max(0, lower), upper=upper)
 df[col_max] = df[col_max].clip(lower=max(0, lower), upper=upper * 1.5)

 return df
```

> **Cap thay vì xóa** để không mất mẫu có text description tốt.

# # # 3.3 Xử lý Null / Missing Values

```python
# Chiến lược cho từng field
NULL_STRATEGY = {
# Text fields – điền chuỗi rỗng, sẽ bị filter ở word_count
 'description': '',
 'requirements': '',
 'benefits': '',

# Structured – điền mode (giá trị phổ biến nhất)
 'industry': df['industry'].mode()[0],
 'job_type': 'Toàn thời gian',
 'location': 'Hồ Chí Minh',

# Numeric – điền median
 'salary_min': df['salary_min'].median(),
 'salary_max': df['salary_max'].median(),
 'experience_required': df['experience_required'].median(),
}

df = df.fillna(NULL_STRATEGY)
```

# # # 3.4 Gộp Rare Categories

```python
def merge_rare_categories(df, col, threshold=0.001):
 """Gộp categories xuất hiện < 0.1% tổng dataset vào nhóm 'Khác'"""
 freq = df[col].value_counts(normalize=True)
 rare = freq[freq < threshold].index
 df[col] = df[col].apply(lambda x: 'Khác' if x in rare else x)
 return df

# Áp dụng cho industry và location
df = merge_rare_categories(df, 'industry', threshold=0.001)
df = merge_rare_categories(df, 'location', threshold=0.001)
```

---

# # BƯỚC 4 – Encode Structured Fields

# # # 4.1 Ordinal Encoding (thứ tự có nghĩa)

```python
EXPERIENCE_MAP = {
 'Không yêu cầu': 0, 'Dưới 1 năm': 1,
 '1-2 năm': 2, '2-3 năm': 3,
 '3-5 năm': 4, 'Trên 5 năm': 5
}

EDUCATION_MAP = {
 'Không yêu cầu': 0, 'Trung cấp': 1,
 'Cao đẳng': 2, 'Đại học': 3, 'Trên đại học': 4
}

COMPANY_SIZE_MAP = {
 'Dưới 10': 0, '10-50': 1, '50-100': 2,
 '100-300': 3, '300-1000': 4, 'Trên 1000': 5
}
```

# # # 4.2 One-Hot Encoding (không có thứ tự)

```python
# Áp dụng cho: industry, location, job_type
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
structured_encoded = ohe.fit_transform(df[['industry', 'location', 'job_type']])
# Kỳ vọng ~55 chiều sau OHE
```

# # # 4.3 Normalize Salary

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['salary_min_norm', 'salary_max_norm']] = scaler.fit_transform(
 df[['salary_min', 'salary_max']]
)
```

**Tổng structured features:** ~60 chiều

---

# # BƯỚC 5 – Vector hóa Text Fields

# # # Phương án A – PhoBERT (Khuyến nghị – chất lượng cao)

```python
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = 'vinai/phobert-base-v2' # Hoặc phobert-large nếu đủ RAM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text: str, max_length: int = 256) -> np.ndarray:
 inputs = tokenizer(
 text,
 return_tensors='pt',
 max_length=max_length,
 truncation=True,
 padding='max_length'
 )
 with torch.no_grad():
 outputs = model(**inputs)
# Mean pooling – lấy trung bình hidden states
 embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
 return embedding # shape: (768,)

# Batch processing để tăng tốc
BATCH_SIZE = 32
embeddings = []
for i in range(0, len(df), BATCH_SIZE):
 batch = df['text_combined'].iloc[i:i+BATCH_SIZE].tolist()
# ... xử lý batch
```

> **Lưu ý RAM:** PhoBERT-base cần ~500MB GPU. Nếu CPU-only, dùng batch nhỏ
> và thêm `model.eval()` để tiết kiệm bộ nhớ.

# # # Phương án B – TF-IDF (Nhẹ hơn – dùng khi tài nguyên hạn chế)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(
 max_features=10000,
 min_df=5, # Bỏ từ xuất hiện < 5 lần
 max_df=0.85, # Bỏ từ xuất hiện > 85% docs (quá phổ biến)
 ngram_range=(1, 2) # Unigram + Bigram
)
tfidf_matrix = tfidf.fit_transform(df['text_combined'])

# Bắt buộc giảm chiều ngay sau TF-IDF
svd = TruncatedSVD(n_components=300, random_state=42)
text_vectors = svd.fit_transform(tfidf_matrix)
```

**So sánh:**

| | PhoBERT | TF-IDF + SVD |
|---|---|---|
| Chất lượng embedding | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Hiểu ngữ cảnh | Có | Không |
| Tốc độ | Chậm | Nhanh |
| RAM cần | ~2–4GB | ~500MB |
| Chiều output | 768 | 300 (sau SVD) |

---

# # BƯỚC 6 – Xử lý Outlier Tầng 2 (Vector Space)

> Thực hiện TRƯỚC khi UMAP, TRÊN full feature vector (~828 chiều).

# # # 6.1 Isolation Forest (Khuyến nghị chính)

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
 contamination=0.04, # Kỳ vọng ~4% outlier trong job data thực tế
 n_estimators=200,
 random_state=42,
 n_jobs=-1
)

outlier_labels = iso_forest.fit_predict(full_feature_matrix)
# -1 = outlier, 1 = inlier

# Tách outlier ra để phân tích riêng (không xóa hẳn)
df_inliers = df[outlier_labels == 1].copy()
df_outliers = df[outlier_labels == -1].copy()

print(f"Inliers: {len(df_inliers)} | Outliers: {len(df_outliers)}")
```

# # # 6.2 LOF – Local Outlier Factor (So sánh / Validation)

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
 n_neighbors=20,
 contamination=0.04,
 n_jobs=-1
)
lof_labels = lof.fit_predict(full_feature_matrix)
```

# # # 6.3 Quyết định xử lý outlier

```python
# Chiến lược bảo thủ: chỉ loại khi CẢ HAI đồng thuận là outlier
final_outlier_mask = (outlier_labels == -1) & (lof_labels == -1)

df_clean = df[~final_outlier_mask].copy()
print(f"Loại {final_outlier_mask.sum()} outliers ({final_outlier_mask.mean()*100:.1f}%)")
```

---

# # BƯỚC 7 – Kết hợp Feature Vector

```python
import numpy as np

# text_vectors: (N, 768) từ PhoBERT
# structured_encoded: (N, ~57) từ OHE + ordinal + salary

full_feature_matrix = np.hstack([
 text_vectors, # 768 chiều – trọng số cao nhất
 structured_encoded, # ~57 chiều
 ordinal_features, # 3 chiều (experience, education, company_size)
 salary_features, # 2 chiều (salary_min_norm, salary_max_norm)
])
# Shape: (N, ~830)

# Chuẩn hóa toàn bộ matrix trước UMAP
from sklearn.preprocessing import StandardScaler
scaler_final = StandardScaler()
full_feature_matrix = scaler_final.fit_transform(full_feature_matrix)
```

---

# # BƯỚC 8 – Giảm chiều (UMAP)

```python
import umap

# Cho Clustering (giảm về 15–30 chiều)
reducer_cluster = umap.UMAP(
 n_components=20, # Chiều đầu ra cho clustering
 n_neighbors=15, # Cân bằng local vs global structure
 min_dist=0.1, # Cho phép cluster gần nhau
 metric='cosine', # Tốt hơn euclidean với text vectors
 random_state=42
)
embeddings_cluster = reducer_cluster.fit_transform(full_feature_matrix)

# Cho Visualization (giảm về 2 chiều)
reducer_viz = umap.UMAP(
 n_components=2,
 n_neighbors=15,
 min_dist=0.05, # Compact hơn để nhìn rõ cluster
 metric='cosine',
 random_state=42
)
embeddings_2d = reducer_viz.fit_transform(full_feature_matrix)
```

> **Lưu ý:** Fit UMAP một lần trên toàn bộ data, lưu `reducer` object lại
> để transform data mới sau này.

---

# # Checklist kiểm tra chất lượng sau pipeline

```
□ Số mẫu còn lại >= 90% ban đầu
□ Không có NaN trong full_feature_matrix
□ Text combined trung bình >= 100 từ
□ Salary đã được normalize về [0, 1]
□ Rare categories đã được gộp (< 20 unique values mỗi OHE field)
□ Outlier tỷ lệ loại: 3–8% (nếu > 10% → kiểm tra lại ngưỡng)
□ embeddings_cluster shape: (N_clean, 20)
□ embeddings_2d shape: (N_clean, 2)
```

---

# # Chiều dữ liệu tóm tắt

| Giai đoạn | Số chiều | Số mẫu |
|---|---|---|
| Raw data | 18–22 cols | N |
| Sau drop fields | 11–13 cols | N |
| Sau encode structured | ~60 | N |
| + PhoBERT embedding | ~828 | N |
| Sau Outlier Tầng 2 | ~828 | ~95% N |
| **Sau UMAP (clustering)** | **20** | **~95% N** |
| Sau UMAP (visualization) | 2 | ~95% N |

---

# # Dependencies

```
underthesea>=6.8.0 # Vietnamese NLP tokenizer
pyvi>=0.1.1 # Vietnamese tokenizer (backup)
transformers>=4.35.0 # PhoBERT
torch>=2.0.0 # PyTorch backend
scikit-learn>=1.3.0 # Encoding, IsolationForest, LOF, TF-IDF
umap-learn>=0.5.5 # UMAP dimensionality reduction
numpy>=1.24.0
pandas>=2.0.0
```

```bash
pip install underthesea pyvi transformers torch scikit-learn umap-learn numpy pandas
```

---

# # Các file tham chiếu

- `references/vietnamese_stopwords.txt` – Danh sách stopwords tiếng Việt
- `references/industry_mapping.json` – Mapping tên ngành (chuẩn hóa tên khác nhau về cùng category)
- `references/salary_ranges_vn.json` – Tham chiếu mức lương theo ngành để validate IQR