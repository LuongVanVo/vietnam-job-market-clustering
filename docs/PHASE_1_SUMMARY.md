# Báo cáo Kết quả Giai đoạn 1 (Phase 1 Summary Report)

Tài liệu này tổng hợp toàn bộ các công việc, logic xử lý, kích thước tập dữ liệu và kết quả trực quan hóa đã hoàn thành trong Giai đoạn 1: Thu thập, Chia tách, và Làm sạch dữ liệu của dự án Phân cụm thị trường việc làm Việt Nam.

---

# # Các Tệp tin Đã Được Tạo ra

Hệ thống đã xây dựng và tổ chức cấu trúc dự án hoàn chỉnh gồm các thư mục và tệp tin sau:

1. **Thư mục data (Lưu trữ dữ liệu thô và sạch)**:
 * `raw_data_train.csv` & `raw_data_test.csv`: Phân chia tập Train/Test tỷ lệ 90/10 từ bộ dữ liệu thô Hugging Face.
 * `clean_data_train.csv` & `clean_data_test.csv`: Dữ liệu sau khi đi qua pipeline chuẩn hóa văn bản tiếng Việt, trích xuất thuộc tính số, điền khuyết và capping ngoại lệ lương.
2. **Thư mục notebooks (Mã nguồn tương tác tương thích nộp bài)**:
 * `01_preprocess.ipynb`: Notebook Jupyter được chia tách cell chi tiết, chú thích tiếng Việt đầy đủ. Notebook này đã được thực thi sẵn và nhúng trực tiếp các hình ảnh biểu đồ phân phối cũng như lệnh `display(df.head(10))` sau mỗi bước xử lý. Có tích hợp bộ kiểm tra tệp tin cục bộ để tối ưu hóa thời gian tải dữ liệu.
3. **Thư mục plots (Đồ thị trực quan)**:
 * `word_count_comparison.png`: Phân bố độ dài từ (trước vs sau clean).
 * `salary_comparison.png`: Biểu đồ hộp thể hiện khoảng lương (trước vs sau capping).
4. **Các kịch bản tiện ích ở thư mục gốc**:
 * `preprocess.py`: File chạy tự động toàn bộ quy trình tiền xử lý dữ liệu qua Command Line.
 * `head.py`: Kịch bản Python xem nhanh 10 dòng đầu của dữ liệu sạch.
 * `README.md`: Hướng dẫn vận hành chương trình chi tiết bằng tiếng Việt.

---

# # Logic Tiền Xử Lý Dữ Liệu Chi Tiết

# # # 1. Chuẩn hóa Văn bản Tiếng Việt
* Đưa toàn bộ văn bản về chữ thường (lowercase) để đảm bảo tính đồng nhất.
* Xóa thẻ HTML (`<[^>]+>`), URL, email để loại bỏ rác văn bản.
* Xóa số điện thoại Việt Nam dạng thô (`0...` hoặc `+84...`) nhằm bảo vệ thông tin cá nhân và giảm nhiễu.
* Chỉ giữ lại ký tự chữ cái tiếng Việt có dấu, chữ số và khoảng trắng đơn.

# # # 2. Chuẩn hóa Địa lý và Quy hoạch Địa phương (Location Classification)
* Quy chuẩn hóa các chuỗi địa chỉ thô chứa nhiều nhiễu (số nhà, tên ngõ, đường cụ thể với hơn 233,000 giá trị độc lập) bằng cách trích lọc tỉnh thành/quận huyện lớn thông qua hàm ánh xạ từ khóa địa lý `clean_location`.
* **Tăng tính khách quan:** Thay vì áp dụng bộ lọc thủ công gộp nhóm tần suất dưới 0.1% ở Giai đoạn 1 (vốn mang tính chủ quan), dự án giữ nguyên toàn bộ **4,179 địa phương sạch** để chuyển tiếp cho Phase 2. Toàn bộ quá trình gộp nhóm hiếm sẽ được xử lý tự động và khách quan thông qua tham số `min_frequency=0.005` (0.5%) của `OneHotEncoder` trong không gian đặc trưng.
* **Bảo toàn dữ liệu địa lý:** Việc bỏ qua lọc thủ công giúp tỷ lệ gộp vào nhóm "Khác" giảm mạnh từ **8.3% xuống chỉ còn 3.24% (17,691 dòng)**, giúp bảo toàn hơn **96.76% thông tin địa lý** thực tế.

# # # 3. Trích xuất Thuộc tính Số từ Chuỗi Text Cấu trúc
* **Trích xuất Lương (salary_min_m_vnd & salary_max_m_vnd)**: Phân tích cú pháp các mẫu chuỗi lương dạng triệu đồng, USD, nghìn/tuần, VND tháng và đồng nhất quy đổi về đơn vị **Triệu VND / tháng**.
* **Trích xuất Kinh nghiệm (exp_min_years & exp_max_years)**: Phân tích chuỗi mô tả kinh nghiệm yêu cầu thành số năm tối thiểu và tối đa dạng float/int.

# # # 4. Xử lý Dữ liệu trống (Imputation)
* **Thuộc tính phân loại/văn bản**: Điền các giá trị trống bằng **Yếu vị (Mode)** tính từ tập Train thô (ví dụ: Địa điểm điền mode là *Hồ Chí Minh*, loại công việc điền mode là *Toàn thời gian*).
* **Thuộc tính số (Lương & Kinh nghiệm)**: Điền các ô trống bằng **Trung vị (Median)** tính từ tập Train thô (Lương Min điền 9.0M, Lương Max điền 15.0M, Kinh nghiệm Min/Max điền 3.0 năm).

# # # 5. Loại bỏ & Điều chỉnh Ngoại lệ (Outlier Handling)
* **Ngoại lệ Độ dài Văn bản (Text Length Outliers)**: Đo độ dài từ (`word_count`) của trường text kết hợp. Thực hiện **loại bỏ** các bản ghi có số từ dưới 50 từ (quá ngắn/tin rác) hoặc trên 5000 từ (quá dài/copy-paste sai định dạng).
* **Ngoại lệ Mức lương (Salary Outliers)**: Áp dụng quy tắc IQR trên thuộc tính lương tối thiểu tập Train để xác định biên trên giới hạn ở mức **23.0 Triệu VND/tháng** (ngưỡng `Q3 + 3.0 * IQR`). Thực hiện **capping** (giới hạn đầu lương) các mức lương cực cao về ngưỡng này thay vì xóa hàng để giữ lại văn bản JD tốt phục vụ cho clustering.

---

# # Bảng Thống kê Kích thước & Số liệu Dữ liệu

| Chỉ số thống kê | Dữ liệu Thô (Raw) | Dữ liệu Sạch (Clean) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Kích thước Train** | (546,190, 11) | **(545,805, 17)** | Loại bỏ 385 dòng text ngoại lệ và mở rộng thêm 6 cột số/văn bản kết hợp |
| **Kích thước Test** | (60,688, 11) | **(60,644, 17)** | Loại bỏ 44 dòng text ngoại lệ và mở rộng thêm 6 cột số/văn bản kết hợp |
| **Khoảng số từ (Train)** | 0 đến 7,571 từ | **50 đến 4,092 từ** | Khử các tin cực đoan gây bùng nổ không gian vector |
| **Khoảng lương tối thiểu (Train)** | 0.0M đến 500.0M VND | **1.0M đến 23.0M VND** | Capping các mức lương cực cao theo IQR |
| **Khoảng kinh nghiệm tối thiểu** | N/A | **0.0 đến 15.0 năm** | Điền khuyết và chuẩn hóa số thành công |
| **Tỷ lệ Null các thuộc tính** | Dao động từ 5% - 35% | **0.00%** | Đã được điền khuyết hoàn toàn bằng Mode/Median |
| **Tỷ lệ nhóm địa phương "Khác"** | N/A | **3.24%** | Giảm thiểu tối đa so với 8.3% trước đó, giữ lại 4,179 địa phương chi tiết |
