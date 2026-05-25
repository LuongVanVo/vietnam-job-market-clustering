# Báo cáo Kết quả Giai đoạn 2 (Phase 2 Summary Report)

Tài liệu này tổng hợp toàn bộ các công việc, logic xử lý, kích thước tập dữ liệu và kết quả trực quan hóa đã hoàn thành trong **Giai đoạn 2: Trích xuất Đặc trưng, Vector hóa và Giảm chiều**.

---

# # Các Tệp tin Đã Được Tạo ra

Hệ thống đã xây dựng và tổ chức cấu trúc dự án hoàn chỉnh cho Giai đoạn 2 gồm các thư mục và tệp tin sau:

1. **Thư mục [`data/`](file:///Users/anhnon/vietnam-job-market-clustering/data/) (Ma trận đặc trưng số)**:
 * `clean_data_train_final.csv`: Dữ liệu Train sạch sau cả 2 tầng lọc ngoại lệ (ngữ pháp ở Phase 1 và ngữ nghĩa ở Phase 2).
 * `features_train.npz`: File nén numpy chứa ma trận đặc trưng Train giảm chiều gồm:
 * `features_20d`: Ma trận 20 chiều phục vụ phân cụm (Clustering).
 * `features_2d`: Ma trận 2 chiều phục vụ trực quan hóa (Visualization).
 * `features_test.npz`: File nén numpy chứa ma trận đặc trưng Test giảm chiều (`features_20d` và `features_2d`).
2. **Thư mục [`models/`](file:///Users/anhnon/vietnam-job-market-clustering/models/) (Pipeline tiền xử lý đã huấn luyện)**:
 * `scaler_num.pkl`: Bộ MinMaxScaler cho các thuộc tính số.
 * `ohe.pkl`: Bộ One-Hot Encoder cho các thuộc tính phân loại.
 * `tfidf.pkl`: Bộ Vector hóa TF-IDF mô tả công việc.
 * `svd.pkl`: Bộ giảm chiều TruncatedSVD cho văn bản (100 chiều).
 * `iso_forest.pkl`: Bộ phát hiện ngoại lệ Isolation Forest.
 * `scaler_final.pkl`: Bộ StandardScaler chuẩn hóa cuối cùng.
 * `umap_cluster.pkl`: Mô hình UMAP giảm về 20 chiều.
 * `umap_viz.pkl`: Mô hình UMAP giảm về 2 chiều.
3. **Thư mục [`notebooks/`](file:///Users/anhnon/vietnam-job-market-clustering/notebooks/) (Notebook nộp bài)**:
 * `feature_engineering.ipynb`: Notebook Jupyter được chia tách cell chi tiết, chú thích tiếng Việt 100%. Notebook này đã được thực thi sẵn, hiển thị biểu đồ phân bố 2D UMAP và bảng kết quả thống kê.
4. **Thư mục [`plots/`](file:///Users/anhnon/vietnam-job-market-clustering/plots/) (Đồ thị trực quan)**:
 * `feature_distribution_2d.png`: Bản đồ phân bố 2D UMAP của 20,000 tin tuyển dụng ngẫu nhiên, tô màu theo mức lương để kiểm chứng tính phân tách hình học.
5. **Tệp chạy tự động ở thư mục gốc**:
 * `feature_engineering.py`: File chạy tự động toàn bộ quy trình tiền xử lý và trích xuất đặc trưng Phase 2 qua Command Line.

---

# # Quy trình Trích xuất Đặc trưng & Giảm chiều

# # # 1. Mã hóa Đặc trưng Cấu trúc
* **Ordinal Encoding**: Trường học vấn `education_level` được chuyển đổi sang số bậc từ 0 (Không yêu cầu) đến 5 (Đại học/Kỹ sư) để giữ lại quan hệ thứ tự.
* **One-Hot Encoding**: Các trường phân loại (`location`, `job_type`, `job_industry`, `job_position`) được mã hóa OHE. Thiết lập `min_frequency=0.005` giúp tự động gộp các nhóm hiếm gặp (tần suất xuất hiện < 0.5% trong 540k tin) vào một danh mục chung để tránh bùng nổ số lượng cột và giảm nhiễu.
* **MinMaxScaler**: Chuẩn hóa lương và số năm kinh nghiệm về khoảng $[0, 1]$.

# # # 2. Vector hóa Văn bản (TF-IDF + TruncatedSVD)
* Do giới hạn về thời gian xử lý của CPU (sẽ mất ~20 tiếng chạy PhoBERT cho 540k dòng), hệ thống sử dụng phương án tối ưu: **TF-IDF (10,000 đặc trưng)** kết hợp **TruncatedSVD** nén về **100 chiều** đặc trưng ngữ nghĩa tiềm ẩn (LSA). Phương án này xử lý toàn bộ 540k dòng chỉ mất **2 phút** và mang lại hiệu quả gom cụm từ vựng rất cao.

# # # 3. Khử Ngoại lệ tầng 2 (Isolation Forest)
* Ghép nối đặc trưng văn bản và đặc trưng cấu trúc thành ma trận 169 chiều.
* Áp dụng **Isolation Forest** với tỷ lệ `contamination=0.04` lọc bỏ các tin tuyển dụng dị biệt, lệch chuẩn trong không gian đặc trưng chung (loại bỏ **21,832 dòng ngoại lệ**).

# # # 4. Giảm chiều UMAP tối ưu
* **Giải pháp tối ưu hóa bộ nhớ**: Lấy mẫu đại diện **50,000 dòng** ngẫu nhiên từ tập Train sạch để huấn luyện mô hình UMAP (học hàm chiếu phi tuyến tính).
* **Chiếu toàn bộ dữ liệu**: Dùng mô hình đã học để biến đổi (`transform`) **toàn bộ 100% dữ liệu** (523,973 dòng Train và 60,644 dòng Test) về không gian **20 chiều** (phục vụ phân cụm) và **2 chiều** (phục vụ vẽ biểu đồ). Việc này đảm bảo tính toán nhanh, không bị lỗi tràn bộ nhớ (Out-Of-Memory) nhưng vẫn đảm bảo **phân cụm trên toàn bộ dữ liệu**.

---

# # Kích thước Ma trận Đặc trưng Đầu ra

* `train_20d` (UMAP 20D cho Phân cụm): **(523,973, 20)**
* `train_2d` (UMAP 2D cho Trực quan): **(523,973, 2)**
* `test_20d` (UMAP 20D cho Phân cụm): **(60,644, 20)**
* `test_2d` (UMAP 2D cho Trực quan) : **(60,644, 2)**
* **Trạng thái kiểm tra giá trị khuyết thiếu (NaN)**: Không chứa NaN ở cả hai tập.
