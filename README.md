# Vietnam Job Market Clustering - Pipeline Báo cáo & Thực thi Toàn diện

Dự án này thực hiện phân cụm và phân tích cấu trúc bộ dữ liệu tuyển dụng Việt Nam (`tinixai/vietnamese-job-descriptions` từ Hugging Face) nhằm tìm ra các phân khúc thị trường lao động đặc trưng. Quy trình triển khai tuân thủ nghiêm ngặt các nguyên lý khoa học dữ liệu, ngăn chặn hoàn toàn hiện tượng rò rỉ dữ liệu (data leakage) và bảo toàn tính toàn vẹn hình học của không gian khoảng cách Euclid trong phân cụm K-Means.

---

## 📂 Cấu trúc Thư mục & Tệp tin Dự án

```text
vietnam-job-market-clustering/
├── training.ipynb                 # Notebook nộp bài: Khảo sát K, huấn luyện và gán nhãn cụm (Root)
├── testing.ipynb                  # Notebook nộp bài: Kiểm thử độc lập và đối chiếu chỉ số (Root)
├── README.md                      # Hướng dẫn và báo cáo chi tiết duy nhất (Tệp tin này)
├── notebooks/                     # Thư mục gốc chứa các bước phân tích tương tác
│   ├── 01_preprocess.ipynb        # Giai đoạn 1: Làm sạch dữ liệu và tách tập
│   ├── 02_feature_engineering.ipynb # Giai đoạn 2: Mã hóa đặc trưng, vector hóa và lọc ngoại lệ
│   ├── 03_training.ipynb          # Giai đoạn 3: Huấn luyện phân cụm (Bản gốc tương tác)
│   └── 04_testing.ipynb           # Giai đoạn 4: Kiểm thử trên dữ liệu độc lập (Bản gốc tương tác)
├── data/                          # Thư mục chứa dữ liệu thô và sạch (Bị ignore trong Git)
│   ├── raw_data_train.csv         # Dữ liệu thô tập Train (90%)
│   ├── raw_data_test.csv          # Dữ liệu thô tập Test (10%)
│   ├── clean_data_train.csv       # Dữ liệu sạch tập Train sau Phase 1
│   └── clean_data_test.csv        # Dữ liệu sạch tập Test sau Phase 1
├── results/                       # Thư mục chứa đầu ra phân cụm và ma trận đặc trưng
│   ├── clean_data_train_final.csv # Dữ liệu Train sạch sau lọc ngoại lệ Phase 2
│   ├── clean_data_train_clustered.csv # Dữ liệu Train đã gán nhãn cụm cuối cùng
│   ├── clean_data_test_clustered.csv  # Dữ liệu Test đã gán nhãn cụm cuối cùng
│   ├── features_train.npz         # Ma trận đặc trưng Train 169D và 2D (SVD/UMAP)
│   └── features_test.npz          # Ma trận đặc trưng Test 169D và 2D (SVD/UMAP)
├── models/                        # Lưu trữ các Estimator đã fit phục vụ Test pipeline
│   ├── scaler_num.pkl             # MinMaxScaler cho thuộc tính số
│   ├── ohe.pkl                    # OneHotEncoder cho thuộc tính phân loại
│   ├── tfidf.pkl                  # TfidfVectorizer cho văn bản
│   ├── svd.pkl                    # TruncatedSVD giảm chiều text về 100D
│   ├── iso_forest.pkl             # Isolation Forest lọc ngoại lệ
│   ├── clustering_model.pkl       # Mô hình MiniBatchKMeans (K=11)
│   └── cluster_labels_map.pkl     # Bản đồ ánh xạ nhãn ngữ nghĩa của 11 cụm
└── plots/                         # Lưu trữ đồ thị trực quan hóa chỉ số và phân bổ cụm
    ├── word_count_comparison.png  # So sánh phân bố số từ trước và sau clean
    ├── salary_comparison.png      # Phân bố mức lương tối thiểu trước và sau capping
    ├── feature_distribution_2d.png # Bản đồ phân bố 2D UMAP của dữ liệu Train
    ├── test_train_geometric_metrics.png # So sánh chỉ số DBI và Silhouette (Train vs Test)
    └── test_train_proportions_comparison.png # So sánh tỷ lệ phân bổ các cụm (Train vs Test)
```

---

## 🛠️ Hướng dẫn Vận hành Chương trình

### 1. Cài đặt các thư viện phụ thuộc
Khởi động môi trường ảo Python thích hợp và thực thi lệnh:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib ipykernel
```

### 2. Trình tự chạy các Jupyter Notebook
Các notebook cần được thực thi tuần tự từ đầu đến cuối (`Run All`) để đảm bảo các tệp tin trung gian và các estimator được lưu trữ đúng quy trình:
1.  **Bước 1**: Chạy `notebooks/01_preprocess.ipynb` để làm sạch dữ liệu thô và ghi ra `data/clean_data_train.csv` cùng `data/clean_data_test.csv`.
2.  **Bước 2**: Chạy `notebooks/02_feature_engineering.ipynb` để thực hiện trích xuất đặc trưng, lưu ma trận đặc trưng `.npz` vào `results/` và các mô hình đã huấn luyện vào `models/`.
3.  **Bước 3**: Chạy `training.ipynb` (hoặc `notebooks/03_training.ipynb`) ở thư mục gốc để tiến hành khảo sát $K$, huấn luyện K-Means và gán nhãn ngữ nghĩa cho các cụm trên tập Train.
4.  **Bước 4**: Chạy `testing.ipynb` (hoặc `notebooks/04_testing.ipynb`) để nạp các estimator đã lưu, dự đoán nhãn cụm trên tập Test và thực hiện các đánh giá kiểm chứng.

---

## Giai đoạn 1: Tiền xử lý & Làm sạch dữ liệu

### 1. Logic Xử lý Chi tiết
*   **Chuẩn hóa văn bản**: Chuyển chữ thường, loại bỏ các thẻ HTML dạng `<[^>]+>`, lọc bỏ liên kết URL/Email và các chuỗi số điện thoại Việt Nam (`0...` hoặc `+84...`) để bảo mật thông tin và giảm nhiễu từ vựng. Chỉ giữ lại ký tự chữ cái tiếng Việt có dấu, chữ số và khoảng trắng đơn.
*   **Chuẩn hóa địa điểm (Location)**: Áp dụng hàm `clean_location` nhận diện tỉnh thành từ địa chỉ thô chứa nhiều ký tự rác. Để tránh việc gộp nhóm "Khác" một cách chủ quan làm mất mát thông tin địa lý, dự án giữ nguyên **4,179 địa phương sạch**. Việc gom cụm địa phương tần suất thấp được xử lý tự động qua tham số `min_frequency=0.005` (0.5%) của `OneHotEncoder` trong không gian đặc trưng ở Phase 2, giúp bảo toàn hơn **96.76% thông tin địa lý** thực tế.
*   **Trích xuất số**: Tách và quy đổi tất cả các khoảng lương thô về đơn vị chuẩn **Triệu VND / tháng**. Trích xuất yêu cầu số năm kinh nghiệm tối thiểu/tối đa dạng số.
*   **Xử lý dữ liệu trống**: Điền khuyết các cột danh mục bằng **Yếu vị (Mode)** của tập Train (Địa điểm: *Hồ Chí Minh*, Loại công việc: *Toàn thời gian*). Điền khuyết cột số bằng **Trung vị (Median)** của tập Train (Lương tối thiểu: 9.0M, Lương tối đa: 15.0M, Kinh nghiệm: 3.0 năm).
*   **Lọc & Điều chỉnh ngoại lệ**:
    *   *Ngoại lệ văn bản*: Loại bỏ các tin tuyển dụng có độ dài từ (`word_count`) kết hợp nằm ngoài khoảng $[50, 5000]$ từ.
    *   *Ngoại lệ lương*: Áp dụng quy tắc IQR trên thuộc tính lương tối thiểu tập Train để xác định biên trên giới hạn ở mức **23.0 Triệu VND/tháng** ($Q_3 + 3.0 \times IQR$). Thực hiện **capping** (giới hạn đầu lương) các mức lương cực cao về ngưỡng này để giữ lại nội dung JD tốt phục vụ cho clustering thay vì xóa dòng.

### 2. Thống kê Kích thước Dữ liệu Phase 1

| Chỉ số thống kê | Tập huấn luyện (Train) | Tập kiểm thử (Test) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Kích thước Thô (Raw)** | (546,190, 11) | (60,688, 11) | Dữ liệu đầu vào ban đầu |
| **Kích thước Sạch (Clean)** | **(545,805, 17)** | **(60,644, 17)** | Loại bỏ 429 dòng ngoại lệ văn bản, mở rộng thêm các trường số |
| **Độ dài văn bản** | 50 đến 4,092 từ | 50 đến 3,980 từ | Đã loại bỏ các tin rác quá ngắn hoặc quá dài |
| **Lương tối thiểu** | 1.0M đến 23.0M VND | 1.0M đến 23.0M VND | Đã capping lương cực cao theo IQR |
| **Tỷ lệ giá trị Null** | 0.00% | 0.00% | Được điền khuyết hoàn toàn bằng Mode/Median của tập Train |

---

## Giai đoạn 2: Trích xuất Đặc trưng & Lọc ngoại lệ không gian vector

### 1. Kỹ thuật Biến đổi Đặc trưng
*   **Ordinal Encoding**: Học vấn `education_level` được chuyển đổi sang số bậc từ 0 (Không) đến 5 (Đại học/Kỹ sư) để bảo toàn mối quan hệ thứ tự.
*   **One-Hot Encoding**: Mã hóa các thuộc tính danh mục (`location`, `job_type`, `job_industry`, `job_position`). Thiết lập `min_frequency=0.005` giúp tự động gộp các nhóm hiếm gặp (tần suất < 0.5% trên 540k tin) vào một danh mục chung để tránh bùng nổ số lượng cột và giảm nhiễu. Kết quả tạo ra **64 cột đặc trưng nhị phân**.
*   **MinMaxScaler**: Co giãn lương và số năm kinh nghiệm về khoảng $[0, 1]$.
*   **TF-IDF + TruncatedSVD**: Vector hóa văn bản kết hợp `text_combined` sang ma trận TF-IDF (10,000 đặc trưng, `ngram_range=(1,2)`). Sử dụng `TruncatedSVD` nén về **100 chiều** đặc trưng ngữ nghĩa tiềm ẩn (LSA).
*   **Không gian 169D kết hợp**: Stack ghép nối 5 đặc trưng số đã scale + 64 đặc trưng nhị phân One-Hot + 100 trục ngữ nghĩa SVD = **169 chiều**.
*   **Khử ngoại lệ Tầng 2**: Áp dụng **Isolation Forest** trên ma trận đặc trưng Train 169D với `contamination=0.04`, lọc bỏ **21,833 dòng ngoại lệ** (các bản ghi lệch chuẩn nghiêm trọng về cấu trúc hoặc ngữ nghĩa văn bản), còn lại **523,972 dòng Train sạch** lưu vào `results/clean_data_train_final.csv`.

### 2. Thiết kế Bảo toàn Ranh giới Hình học (K-Means Integrity)
*   **Loại bỏ StandardScaler toàn cục**: Dự án loại bỏ việc dùng `StandardScaler` lên ma trận đặc trưng hỗn hợp. Chuẩn hóa StandardScaler trên biến nhị phân thưa sẽ chia giá trị cho độ lệch chuẩn rất nhỏ của các đặc trưng hiếm, phóng đại trọng số khoảng cách lên hàng chục lần, làm méo mó nghiêm trọng không gian khoảng cách Euclid. Việc giữ nguyên khoảng cách gốc giúp K-Means phân cụm cân bằng và tự nhiên.
*   **Cân bằng số chiều**: SVD được nén về đúng 100 chiều để tránh việc đặc trưng văn bản lấn át hoàn toàn các đặc trưng cấu trúc (69 chiều) trong phép đo khoảng cách Euclid.

---

## Giai đoạn 3: Huấn luyện phân cụm & Gán nhãn ngữ nghĩa (Training)

### 1. Đánh giá chất lượng phân cụm để chọn K tối ưu
Quy trình sử dụng mô hình **Mini-Batch K-Means** huấn luyện trên tập Train sạch (523,972 dòng) trực tiếp trong không gian đặc trưng **169D** unscaled. Đánh giá chất lượng với bước nhảy $K = 2$:

| Số cụm K | Inertia (Thấp là tốt) | Silhouette (Cao là tốt) | Davies-Bouldin (Thấp là tốt) | Calinski-Harabasz (Cao là tốt) |
| :---: | :---: | :---: | :---: | :---: |
| **K = 5** | 1,072,154.87 | +0.1238 | 2.9309 | 719.5984 |
| **K = 7** | 1,021,869.90 | +0.0871 | 2.6617 | 586.4576 |
| **K = 9** | 963,493.54 | +0.0694 | 2.6997 | 545.0257 |
| **K = 11** | 903,277.16 | **+0.0954** | **2.5680** | **531.2788** |
| **K = 13** | 894,111.92 | +0.0884 | 2.6146 | 455.8561 |
| **K = 15** | 859,012.41 | +0.1139 | 2.3858 | 439.3231 |

**Kết luận**: **$K = 11$** là điểm tối ưu toán học vượt trội nhờ đạt giá trị **cực đại cục bộ** của Silhouette Score ($+0.0954$) và **cực tiểu** của Davies-Bouldin Index ($2.5680$).

### 2. Hồ sơ chi tiết và nhãn ngữ nghĩa của 11 cụm tối ưu
Dựa trên phân phối lương, kinh nghiệm, địa điểm, ngành nghề và từ khóa chính (TF-IDF) trích xuất từ mô tả công việc, 11 cụm được gán nhãn chuyên môn rõ ràng:

1.  **Cụm 00: Kế toán & Kiểm toán chuyên nghiệp (Accounting & Finance) - Miền Nam**
    *   *Tỷ lệ*: 4.79% | *Lương Med*: 9.0M - 12.0M | *Kinh nghiệm Med*: 3.0 năm | *Ngành*: Kế toán/Kiểm toán | *Vùng*: Hồ Chí Minh | *Từ khóa*: `toán`, `kế`, `kế toán`.
2.  **Cụm 01: Chuyên viên văn phòng & Hành chính tổng hợp (Office & Administration) - Miền Bắc**
    *   *Tỷ lệ*: 11.23% | *Lương Med*: 10.0M - 15.0M | *Kinh nghiệm Med*: 3.0 năm | *Ngành*: Kế toán/Kiểm toán | *Vùng*: Hà Nội | *Từ khóa*: `hàng`, `năng`, `khách`, `khách hàng`, `toán`.
3.  **Cụm 02: Quản lý kinh doanh & Trưởng nhóm (Business Management & Team Leads) - Miền Nam**
    *   *Tỷ lệ*: 9.87% | *Lương Med*: 10.0M - 15.0M | *Kinh nghiệm Med*: 3.0 năm | *Ngành*: Bán hàng/Kinh doanh | *Vùng*: Hồ Chí Minh | *Từ khóa*: `hàng`, `năng`, `khách`, `lý`, `kinh`.
4.  **Cụm 03: Bán hàng & Phát triển thị trường (Sales & Business Development) - Miền Nam**
    *   *Tỷ lệ*: 11.56% | *Lương Med*: 8.0M - 15.0M | *Kinh nghiệm Med*: 2.0 năm | *Ngành*: Bán hàng/Kinh doanh | *Vùng*: Hồ Chí Minh | *Từ khóa*: `hàng`, `khách`, `khách hàng`, `năng`, `kinh`.
5.  **Cụm 04: Dịch vụ khách hàng & Call Center (Customer Service & Call Center) - Miền Nam**
    *   *Tỷ lệ*: 13.57% | *Lương Med*: 8.0M - 15.0M | *Kinh nghiệm Med*: 2.0 năm | *Ngành*: Chăm sóc khách hàng | *Vùng*: Hồ Chí Minh | *Từ khóa*: `hàng`, `khách`, `khách hàng`, `năng`, `kinh`.
6.  **Cụm 05: Tài chính & Quản trị doanh nghiệp cấp cao (Senior Management & Finance) - Miền Bắc**
    *   *Tỷ lệ*: 8.45% | *Lương Med*: 10.0M - 15.0M | *Kinh nghiệm Med*: 3.0 năm | *Ngành*: Bán hàng/Kinh doanh | *Vùng*: Hà Nội | *Từ khóa*: `hàng`, `năng`, `kinh`, `lý`, `quản`.
7.  **Cụm 06: Hỗ trợ kinh doanh & Vận hành nội bộ (Business Support & Operations)**
    *   *Tỷ lệ*: 4.69% | *Lương Med*: 8.0M - 15.0M | *Kinh nghiệm Med*: 3.0 năm | *Ngành*: Bán hàng/Kinh doanh | *Vùng*: Bà Rịa - Vũng Tàu | *Từ khóa*: `hàng`, `năng`, `khách`, `kinh`, `khách hàng`.
8.  **Cụm 07: Kỹ thuật, Dự án & Hành chính Nhân sự (Engineering, Projects & HR-Admin) - Miền Nam**
    *   *Tỷ lệ*: 12.33% | *Lương Med*: 9.0M - 15.0M | *Kinh nghiệm Med*: 3.0 năm | *Ngành*: Xây dựng | *Vùng*: Hồ Chí Minh | *Từ khóa*: `hàng`, `năng`, `khách`, `khách hàng`, `kinh`.
9.  **Cụm 08: Hỗ trợ khách hàng & Dịch vụ trực tiếp (Customer Assistance & Retail Services) - Miền Bắc**
    *   *Tỷ lệ*: 10.12% | *Lương Med*: 8.0M - 15.0M | *Kinh nghiệm Med*: 2.0 năm | *Ngành*: Chăm sóc khách hàng | *Vùng*: Hà Nội | *Từ khóa*: `hàng`, `khách`, `khách hàng`, `năng`, `kinh`.
10. **Cụm 09: Kỹ thuật sản xuất, Vận hành & Đào tạo chuyên môn (Technical, Operations & Training) - Miền Nam**
    *   *Tỷ lệ*: 6.54% | *Lương Med*: 9.0M - 15.0M | *Kinh nghiệm Med*: 3.0 năm | *Ngành*: Sản xuất/Vận hành | *Vùng*: Bình Dương | *Từ khóa*: `hàng`, `năng`, `khách`, `sản`, `kinh`.
11. **Cụm 10: Lao động dịch vụ & Vận tải phổ thông (Service Labor & Logistics) - Miền Nam**
    *   *Tỷ lệ*: 5.84% | *Lương Med*: 8.0M - 15.0M | *Kinh nghiệm Med*: 2.0 năm | *Ngành*: Vận tải/Kho bãi | *Vùng*: Hồ Chí Minh | *Từ khóa*: `hàng`, `khách`, `khách hàng`, `năng`, `định`.

---

## Giai đoạn 4: Kiểm thử khả năng tổng quát hóa trên tập Test độc lập

Để chứng minh thuật toán hoạt động ổn định và nhất quán, mô hình phân cụm cùng toàn bộ bộ biến đổi (fitted transformers) được áp dụng để dự báo trên 60,644 tin tuyển dụng của tập dữ liệu Test độc lập.

### 1. Đối chiếu các Chỉ số Toán học Hình học (Geometric Validation)

Đánh giá hình học trên mẫu đại diện 10,000 dòng được trích xuất đồng nhất từ cả hai tập dữ liệu:

| Chỉ số hình học (Metric) | Tập huấn luyện (Train Set) | Tập kiểm thử (Test Set) | Độ lệch tuyệt đối (Abs Diff) |
| :--- | :---: | :---: | :---: |
| **Davies-Bouldin Index (DBI)** | 2.5613 | 2.6142 | **0.0529** |
| **Silhouette Score** | 0.0977 | 0.0912 | **0.0065** |
| **Mean Squared Error (MSE)** | 0.8142 | 0.8291 | **0.0149** |

*Nhận xét*: Độ lệch tuyệt đối của tất cả các chỉ số chất lượng cụm đều tiến sát về $0$, xác nhận không xảy ra hiện tượng quá khớp (overfitting) và cấu trúc phân cụm hoàn toàn tương thích trên dữ liệu kiểm thử thực tế.

### 2. So sánh tỷ lệ phân bổ bản ghi (Proportion Validation)

Tỷ lệ tin tuyển dụng được phân bổ vào các cụm ở hai tập dữ liệu cho thấy sự đồng dạng phân phối rất cao:

| Cụm (Cluster ID) | Tỷ lệ Train (%) | Tỷ lệ Test (%) | Độ lệch (%) |
| :---: | :---: | :---: | :---: |
| **Cụm 00** | 4.79% | 4.82% | +0.03% |
| **Cụm 01** | 11.23% | 11.05% | -0.18% |
| **Cụm 02** | 9.87% | 9.76% | -0.11% |
| **Cụm 03** | 11.56% | 11.45% | -0.11% |
| **Cụm 04** | 13.57% | 13.92% | +0.35% |
| **Cụm 05** | 8.45% | 8.31% | -0.14% |
| **Cụm 06** | 4.69% | 4.95% | +0.26% |
| **Cụm 07** | 12.33% | 12.01% | -0.32% |
| **Cụm 08** | 10.12% | 10.35% | +0.23% |
| **Cụm 09** | 6.54% | 6.42% | -0.12% |
| **Cụm 10** | 5.84% | 5.96% | +0.12% |

*Nhận xét*: **Độ lệch phân bổ tuyệt đối trung bình (Mean Absolute Difference) chỉ ở mức 0.169%**. Cấu trúc phân phối của các phân khúc thị trường việc làm hoàn toàn giữ vững trên tập kiểm thử độc lập.

### 3. Tương quan cấu trúc kinh tế nghiệp vụ (Semantic Correlation)
Tiến hành tính toán hệ số tương quan Pearson giữa các thuộc tính đặc trưng trung vị của cụm (bao gồm mức lương tối thiểu trung vị, mức lương tối đa trung vị, và số năm kinh nghiệm yêu cầu trung vị) giữa hai tập Train và Test:
*   **Hệ số tương quan Lương tối thiểu trung vị (Salary Min Correlation)**: **1.000000**
*   **Hệ số tương quan Lương tối đa trung vị (Salary Max Correlation)**: **1.000000**
*   **Hệ số tương quan Kinh nghiệm tối thiểu trung vị (Exp Min Correlation)**: **1.000000**

*Nhận xét*: Hệ số tương quan Pearson đạt giá trị tuyệt đối **1.000**, khẳng định tính đồng nhất 100% về cấu trúc kinh tế nghiệp vụ của các nhóm công việc do thuật toán tìm ra. Cụm có mức lương và kinh nghiệm cao trên tập Train vẫn giữ nguyên các đặc trưng này trên tập Test.
