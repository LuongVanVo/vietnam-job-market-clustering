# Giai đoạn 2 — Chuẩn bị môi trường, cấu trúc thư mục, tải dataset và chia raw train/test

## 1. Mục tiêu của giai đoạn

Giai đoạn 2 có mục tiêu chuẩn bị nền tảng kỹ thuật cho toàn bộ dự án phân cụm tin tuyển dụng. Ở giai đoạn này, dữ liệu được tải từ Hugging Face, kiểm tra sơ bộ cấu trúc, sau đó chia thành tập huấn luyện và tập kiểm thử theo đúng yêu cầu mới của đề bài.

Kết quả quan trọng nhất của giai đoạn này là tạo ra hai file dữ liệu thô:

```text
raw_data_train.csv
raw_data_test.csv
```

Hai file này sẽ được sử dụng làm đầu vào cho các giai đoạn tiếp theo như làm sạch dữ liệu, phân tích khám phá dữ liệu, trích xuất đặc trưng và phân cụm.

---

## 2. Dataset sử dụng

Dataset được sử dụng là:

```text
tinixai/vietnamese-job-descriptions
```

Nguồn dữ liệu được lấy từ Hugging Face. Đây là bộ dữ liệu tin tuyển dụng tại Việt Nam, gồm các thông tin dạng bảng và văn bản.

Các cột dữ liệu chính dự kiến bao gồm:

```text
job_title
company_name
salary
location
job_type
job_industry
experience_level
education_level
job_position
job_description
benefits
requirements
year
```

Các trường này phù hợp với đề tài vì chúng phản ánh đầy đủ ba nhóm thông tin chính:

| Nhóm thông tin | Các cột liên quan | Vai trò |
|---|---|---|
| Mô tả công việc | `job_title`, `job_description` | Xác định bản chất công việc |
| Yêu cầu ứng viên | `requirements`, `experience_level`, `education_level` | Xác định kỹ năng, kinh nghiệm và trình độ yêu cầu |
| Đặc điểm tuyển dụng | `salary`, `location`, `job_type`, `job_industry`, `job_position`, `company_name`, `benefits`, `year` | Bổ sung bối cảnh tuyển dụng |

---

## 3. Ý nghĩa của raw data

Trong dự án này, raw data được hiểu là dữ liệu thô sau khi tải từ Hugging Face và chia train/test, nhưng chưa qua các bước làm sạch hoặc biến đổi đặc trưng.

Cụ thể, raw data chưa xử lý:

```text
missing value
duplicate
chuẩn hóa văn bản
parse salary
encoding categorical features
TF-IDF
TruncatedSVD
clustering
```

Quy trình tạo raw data:

```text
Hugging Face dataset
→ Load toàn bộ dữ liệu
→ Kiểm tra schema
→ Chia 90% train và 10% test
→ Lưu raw_data_train.csv
→ Lưu raw_data_test.csv
```

---

## 4. Lý do chia train/test 90/10

Theo yêu cầu mới của giảng viên, dữ liệu cần được chia như sau:

```text
Train set: 90%
Test set: 10%
```

Tập train và test không được trùng nhau.

Trong bài toán phân cụm, không có biến mục tiêu `y` như bài toán phân loại hoặc hồi quy. Tuy nhiên, việc chia train/test vẫn có ý nghĩa:

| Tập dữ liệu | Vai trò |
|---|---|
| Train set | Dùng để fit các bước xử lý đặc trưng và mô hình phân cụm |
| Test set | Dùng để kiểm tra khả năng gán cụm cho dữ liệu mới |
| Test set | Dùng để chọn ngẫu nhiên 10 mẫu demo kết quả phân cụm |

Cách giải thích trong báo cáo:

> Trong bài toán phân cụm, tập huấn luyện được sử dụng để xây dựng mô hình phân cụm và các bước xử lý đặc trưng. Tập kiểm thử được giữ riêng nhằm kiểm tra khả năng gán cụm cho các tin tuyển dụng mới, đồng thời phục vụ phần demo kết quả phân cụm trên 10 mẫu ngẫu nhiên.

---

## 5. Cấu trúc thư mục dự án

Cấu trúc thư mục đề xuất:

```text
STTnhom - Phan cum tin tuyen dung tai Viet Nam/
│
├── data/
│   ├── raw/
│   │   ├── raw_data_train.csv
│   │   └── raw_data_test.csv
│   │
│   ├── clean/
│   │   ├── clean_data_train.csv
│   │   └── clean_data_test.csv
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_project_definition_clustering_jobs.ipynb
│   ├── 02_download_and_split_data.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_eda_visualization.ipynb
│   ├── 05_feature_engineering_for_clustering.ipynb
│   ├── 06_clustering_modeling.ipynb
│   └── 07_cluster_interpretation_and_demo.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── metrics/
│
├── models/
├── report/
├── slides/
├── README.md
└── requirements.txt
```

Ý nghĩa các thư mục:

| Thư mục | Vai trò |
|---|---|
| `data/raw/` | Lưu dữ liệu thô sau khi chia train/test |
| `data/clean/` | Lưu dữ liệu sau khi làm sạch |
| `data/processed/` | Lưu dữ liệu sau feature engineering |
| `notebooks/` | Lưu các notebook triển khai |
| `outputs/figures/` | Lưu biểu đồ |
| `outputs/tables/` | Lưu bảng thống kê |
| `outputs/metrics/` | Lưu kết quả đánh giá mô hình phân cụm |
| `models/` | Lưu các model/vectorizer/pipeline |
| `report/` | Lưu báo cáo |
| `slides/` | Lưu slide thuyết trình |

---

## 6. Các bước thực hiện trong notebook giai đoạn 2

Notebook của giai đoạn này nên đặt tên:

```text
02_download_and_split_data.ipynb
```

Các bước chính trong notebook:

```text
1. Import thư viện cần thiết
2. Khai báo thông tin dự án và dataset
3. Tạo cấu trúc thư mục
4. Tải dataset từ Hugging Face
5. Kiểm tra danh sách cột
6. Chuyển dữ liệu sang pandas DataFrame
7. Kiểm tra số dòng, số cột
8. Kiểm tra missing value sơ bộ
9. Kiểm tra các cột categorical như ngành nghề, địa điểm, năm
10. Chia dữ liệu train/test theo tỷ lệ 90/10
11. Kiểm tra tỷ lệ train/test
12. So sánh phân bố ngành nghề giữa train và test
13. Lưu raw_data_train.csv
14. Lưu raw_data_test.csv
15. Đọc thử lại CSV để xác nhận lưu thành công
16. Lưu metadata và các bảng thống kê giai đoạn 2
```

---

## 7. Các file đầu ra của giai đoạn 2

Sau khi hoàn thành giai đoạn 2, cần có các file:

```text
data/raw/raw_data_train.csv
data/raw/raw_data_test.csv
```

Nên copy thêm ra folder gốc để đảm bảo đúng yêu cầu nộp bài:

```text
raw_data_train.csv
raw_data_test.csv
```

Ngoài ra nên có thêm các file hỗ trợ báo cáo:

```text
outputs/tables/stage_02_metadata.csv
outputs/tables/stage_02_missing_summary_raw.csv
outputs/tables/stage_02_train_test_industry_distribution.csv
outputs/tables/stage_02_train_test_year_distribution.csv
```

---

## 8. Nội dung có thể đưa vào báo cáo

Bộ dữ liệu sử dụng trong đề tài là `tinixai/vietnamese-job-descriptions`, được công bố trên nền tảng Hugging Face. Dataset gồm các tin tuyển dụng tại Việt Nam với các trường thông tin như tiêu đề công việc, tên công ty, mức lương, địa điểm, hình thức làm việc, ngành nghề, kinh nghiệm, trình độ học vấn, vị trí công việc, mô tả công việc, quyền lợi, yêu cầu ứng viên và năm đăng tin.

Theo yêu cầu của đề bài, toàn bộ dataset được sử dụng và chia thành hai tập không trùng nhau: tập huấn luyện chiếm 90% và tập kiểm thử chiếm 10%. Trong bài toán phân cụm, tập huấn luyện được dùng để xây dựng các bước xử lý đặc trưng và mô hình phân cụm, còn tập kiểm thử được sử dụng để kiểm tra khả năng gán cụm cho dữ liệu mới và phục vụ demo kết quả phân cụm trên 10 mẫu ngẫu nhiên.

Dữ liệu ở giai đoạn này được lưu dưới dạng `raw_data_train.csv` và `raw_data_test.csv`. Đây là dữ liệu thô, chưa qua các bước làm sạch như xử lý thiếu dữ liệu, chuẩn hóa văn bản, xử lý mức lương hoặc trích xuất đặc trưng.

---

## 9. Nội dung có thể đưa vào slide

### Slide: Thu thập dữ liệu

```text
Nguồn dữ liệu:
tinixai/vietnamese-job-descriptions trên Hugging Face

Đặc điểm:
- Dữ liệu tin tuyển dụng tại Việt Nam
- Gồm dữ liệu dạng bảng và văn bản
- Các cột chính: job_title, job_description, requirements, salary, location, job_type, job_industry...
```

### Slide: Chia dữ liệu

```text
Theo yêu cầu đề bài:
- Train set: 90%
- Test set: 10%

Vai trò:
- Train set: fit preprocessing và clustering model
- Test set: kiểm tra khả năng gán cụm cho dữ liệu mới
- Test set: chọn 10 mẫu ngẫu nhiên để demo kết quả phân cụm
```

### Slide: Output giai đoạn 2

```text
File dữ liệu thô:
- raw_data_train.csv
- raw_data_test.csv

File thống kê:
- stage_02_metadata.csv
- missing summary
- train/test distribution tables
```

---

## 10. Điểm cần giải thích khi thuyết trình

### Vì sao vẫn chia train/test trong bài phân cụm?

Vì mô hình phân cụm vẫn cần được kiểm tra khả năng áp dụng lên dữ liệu mới. Tập train dùng để học cách biểu diễn và phân cụm dữ liệu, còn tập test dùng để gán cụm cho các tin tuyển dụng chưa được dùng khi xây dựng mô hình.

### Vì sao chia 90/10?

Vì đây là yêu cầu mới của giảng viên. Đồng thời dataset có quy mô lớn nên 10% test vẫn đủ lớn để kiểm tra phân bố và demo.

### Vì sao raw data chưa làm sạch?

Vì cần tách rõ các bước trong quy trình khoa học dữ liệu. Raw data là dữ liệu ban đầu sau khi thu thập/chia tách, còn clean data sẽ được tạo ở giai đoạn làm sạch dữ liệu.

### Vì sao lưu CSV?

Vì đề bài yêu cầu nộp các file dữ liệu thô và dữ liệu sạch. Việc lưu CSV cũng giúp các notebook sau chỉ cần load dữ liệu từ file, không cần tải lại dataset từ Hugging Face.

---

## 11. Checklist hoàn thành giai đoạn 2

```text
[x] Xác định dataset sử dụng
[x] Xác định tỷ lệ chia 90/10
[x] Tạo cấu trúc thư mục dự án
[x] Tạo notebook 02_download_and_split_data.ipynb
[x] Tải dataset từ Hugging Face
[x] Kiểm tra schema và danh sách cột
[x] Kiểm tra missing value sơ bộ
[x] Chia dữ liệu train/test
[x] Lưu raw_data_train.csv
[x] Lưu raw_data_test.csv
[x] Lưu metadata
[x] Lưu bảng thống kê hỗ trợ báo cáo
```

---

## 12. Kết luận giai đoạn 2

Giai đoạn 2 hoàn tất khi dữ liệu đã được tải từ Hugging Face, kiểm tra sơ bộ và chia thành hai tập train/test theo tỷ lệ 90/10. Hai file `raw_data_train.csv` và `raw_data_test.csv` là đầu vào chính cho các giai đoạn tiếp theo. Từ giai đoạn 3, dự án sẽ bắt đầu xử lý chất lượng dữ liệu, bao gồm missing value, dữ liệu trùng lặp, chuẩn hóa văn bản và xử lý mức lương.
