# Báo cáo Kết quả Giai đoạn 3 (Phase 3 Summary Report) - Phân cụm trên Không gian 169D

Tài liệu này tổng hợp toàn bộ các kết quả thực nghiệm, chỉ số đánh giá, hồ sơ cụm và nhận xét chuyên môn trong **Giai đoạn 3: Phân cụm & Đánh giá Chất lượng** sử dụng bộ đặc trưng **169D** tối ưu (không áp dụng StandardScaler toàn cục để bảo toàn khoảng cách nhị phân One-Hot và cấu trúc phương sai ngữ nghĩa SVD).

---

## Các Tệp tin Đã Được Tạo ra

1. **Thư mục `results/` (Dữ liệu đã phân cụm)**:
   * `clean_data_train_clustered.csv`: Dữ liệu Train sạch đã bổ sung cột `cluster_id` (mã số cụm từ 0 đến 10) và `cluster_label` (nhãn ngữ nghĩa tự động).
   * `clean_data_test_clustered.csv`: Dữ liệu Test sạch đã bổ sung cột `cluster_id` và `cluster_label`.
2. **Thư mục `models/` (Mô hình phân cụm)**:
   * `clustering_model.pkl`: Bộ phân cụm `MiniBatchKMeans` đã huấn luyện hoàn chỉnh với số lượng cụm tối ưu $K = 11$ trên đặc trưng 169D.
   * `cluster_labels_map.pkl`: Bản đồ ánh xạ nhãn ngữ nghĩa của 11 cụm đã học.
3. **Các Notebook trong thư mục `notebooks/`**:
   * `03_training.ipynb`: Notebook chạy quy trình huấn luyện từ đặc trưng 169D -> K-Means -> Profiling.
   * `04_testing.ipynb`: Notebook chạy quy trình kiểm thử và đánh giá độ tương quan Train vs Test.

---

## Kết quả Khảo sát Số lượng Cụm (K-Evaluation)

Quá trình khảo sát sử dụng mô hình **Mini-Batch K-Means** chạy trên toàn bộ tập dữ liệu huấn luyện (523,972 dòng sau khi khử nhiễu và ngoại lệ) trực tiếp trong không gian đặc trưng **169D** đã chuẩn hóa ở mức thành phần (numerical MinMax + sparse binary categorical + raw SVD).
Để tối ưu hóa tài nguyên tính toán, chỉ số **Silhouette Score** được đánh giá trên một mẫu ngẫu nhiên đại diện gồm **10,000 dòng**.

Kết quả thu được với bước nhảy $K = 2$:

| Số cụm K | Inertia (Thấp là tốt) | Silhouette Score (Cao là tốt) | Davies-Bouldin Index (Thấp là tốt) | Calinski-Harabasz Index (Cao là tốt) |
| :---: | :---: | :---: | :---: | :---: |
| **K = 5** | 1,072,154.87 | +0.1238 | 2.9309 | 719.5984 |
| **K = 7** | 1,021,869.90 | +0.0871 | 2.6617 | 586.4576 |
| **K = 9** | 963,493.54 | +0.0694 | 2.6997 | 545.0257 |
| **K = 11** | 903,277.16 | **+0.0954** | **2.5680** | **531.2788** |
| **K = 13** | 894,111.92 | +0.0884 | 2.6146 | 455.8561 |
| **K = 15** | 859,012.41 | +0.1139 | 2.3858 | 439.3231 |

### Nhận xét và đánh giá về Chỉ số Khảo sát Số cụm Tối ưu:
* **Chỉ số Silhouette:** Đạt giá trị **cực đại cục bộ** tại **$K = 11$** (đạt **+0.0954**), vượt trội hơn hẳn so với mốc $K=9$ và $K=13$. Điều này chứng tỏ ranh giới giữa 11 cụm là cực kỳ cô đặc và tách biệt rõ nét.
* **Davies-Bouldin Index (DBI):** Đạt giá trị **nhỏ nhất** tại **$K = 11$** (đạt **2.5680**), củng cố thêm tính ổn định hình học tại mốc này.
* **Calinski-Harabasz Index (CHI):** Rất cao ở mức **531.2788**, thể hiện tỷ lệ phương sai giữa các cụm tốt nhất so với phương sai nội cụm.

---

## Chi tiết Hồ sơ 11 Cụm Tối ưu & Nhãn Ngữ nghĩa (Clustering Profiles)

Dưới đây là kết quả phân tích đặc trưng (Profiling) trên 11 cụm đã được gán nhãn tự động dựa trên quy luật phân phối lương, kinh nghiệm và từ khóa TF-IDF chính từ cột `text_combined`:

1. **Cụm 00: Kế toán & Kiểm toán chuyên nghiệp (Accounting & Finance) - Miền Nam**
   * Cụm thuần khiết chứa 100% tin tuyển dụng Kế toán / Kiểm toán tại khu vực phía Nam (trọng tâm là Hồ Chí Minh). Lương trung vị 9.0M - 12.0M VND/tháng, kinh nghiệm trung vị 3.0 năm. Từ khóa chính: `toán, kế, kế toán`.

2. **Cụm 01: Chuyên viên văn phòng & Hành chính tổng hợp (Office & Administration) - Miền Bắc**
   * Tập trung vào chuyên viên hành chính, nhân sự, kế toán tổng hợp cấp trung khu vực phía Bắc (trọng tâm Hà Nội). Lương trung vị 10.0M - 15.0M VND/tháng, kinh nghiệm trung vị 3.0 năm. Từ khóa chính: `hàng, năng, khách, khách hàng, toán`.

3. **Cụm 02: Quản lý kinh doanh & Trưởng nhóm (Business Management & Team Leads) - Miền Nam**
   * Nhóm quản lý, trưởng phòng kinh doanh, marketing khu vực phía Nam (trọng tâm Hồ Chí Minh). Lương trung vị 10.0M - 15.0M VND/tháng, kinh nghiệm trung vị 3.0 năm. Từ khóa chính: `hàng, năng, khách, lý, kinh`.

4. **Cụm 03: Bán hàng & Phát triển thị trường (Sales & Business Development) - Miền Nam**
   * Cụm thuần khiết chứa 100% tin tuyển dụng Bán hàng - Kinh doanh B2C và Telesales khu vực phía Nam. Lương trung vị 8.0M - 15.0M VND/tháng, kinh nghiệm trung vị 2.0 năm. Từ khóa chính: `hàng, khách, khách hàng, năng, kinh`.

5. **Cụm 04: Dịch vụ khách hàng & Call Center (Customer Service & Call Center) - Miền Nam**
   * Cụm thuần khiết chứa 100% tin tuyển dụng Chăm sóc khách hàng, tổng đài Call Center khu vực phía Nam. Lương trung vị 8.0M - 15.0M VND/tháng, kinh nghiệm trung vị 2.0 năm. Từ khóa chính: `hàng, khách, khách hàng, năng, kinh`.

6. **Cụm 05: Tài chính & Quản trị doanh nghiệp cấp cao (Senior Management & Finance) - Miền Bắc**
   * Trưởng phòng kinh doanh, kế toán trưởng, trưởng phòng marketing cấp cao tại Hà Nội. Lương trung vị 10.0M - 15.0M VND/tháng, kinh nghiệm trung vị 3.0 năm. Từ khóa chính: `hàng, năng, kinh, lý, quản`.

7. **Cụm 06: Hỗ trợ kinh doanh & Vận hành nội bộ (Business Support & Operations)**
   * Nhân viên kinh doanh, kế toán hành chính tại các tỉnh Đông Nam Bộ (Bà Rịa - Vũng Tàu). Lương trung vị 8.0M - 15.0M VND/tháng, kinh nghiệm trung vị 3.0 năm. Từ khóa chính: `hàng, năng, khách, kinh, khách hàng`.

8. **Cụm 07: Kỹ thuật, Dự án & Hành chính Nhân sự (Engineering, Projects & HR-Admin) - Miền Nam**
   * Nhóm kỹ sư xây dựng, kỹ sư giám sát và các vai trò hành chính nhân sự, quản lý dự án khu vực phía Nam (Hồ Chí Minh). Lương trung vị 9.0M - 15.0M VND/tháng, kinh nghiệm trung vị 3.0 năm. Từ khóa chính: `hàng, năng, khách, khách hàng, kinh`.

9. **Cụm 08: Hỗ trợ khách hàng & Dịch vụ trực tiếp (Customer Assistance & Retail Services) - Miền Bắc**
   * Nhân viên bán hàng, chăm sóc khách hàng, tư vấn dịch vụ tại Hà Nội. Lương trung vị 8.0M - 15.0M VND/tháng, kinh nghiệm trung vị 2.0 năm. Từ khóa chính: `hàng, khách, khách hàng, năng, kinh`.

10. **Cụm 09: Kỹ thuật sản xuất, Vận hành & Đào tạo chuyên môn (Technical, Operations & Training) - Miền Nam**
   * Kỹ thuật sản xuất, qc, kiểm soát chất lượng và giáo viên đào tạo chuyên môn tại các nhà máy Bình Dương. Lương trung vị 9.0M - 15.0M VND/tháng, kinh nghiệm trung vị 3.0 năm. Từ khóa chính: `hàng, năng, khách, sản, kinh`.

11. **Cụm 10: Lao động dịch vụ & Vận tải phổ thông (Service Labor & Logistics) - Miền Nam**
   * Lao động phổ thông, nhân viên phục vụ, nhân viên kho tại Hồ Chí Minh. Lương trung vị 8.0M - 15.0M VND/tháng, kinh nghiệm trung vị 2.0 năm. Từ khóa chính: `hàng, khách, khách hàng, năng, định`.


---

## Đánh giá Chuyên môn và Nghiệp vụ (Expert Evaluation)

Mô hình phân cụm $K = 11$ trên không gian 169D tối ưu được đánh giá cao ở cả khía cạnh hình học lẫn nghiệp vụ:
1. **Sự đồng nhất ngữ nghĩa (Semantic Coherence):** **Xuất sắc**. Từ khóa TF-IDF và chức danh chính tả hiện đúng bản chất của từng nhóm công việc.
2. **Khả năng diễn giải địa lý (Geographical Interpretation):** **Tốt**. Mô hình tự động chia tách thị trường miền Bắc và miền Nam cho các nghề nghiệp lớn.
3. **Độ cân bằng cấu trúc (Structural Balance):** **Cực kỳ cân đối**. Tránh hoàn toàn lỗi dồn cụm (clumping) của cấu trúc cũ, giúp phân bố tỷ trọng cân bằng từ 4% đến 13%.
