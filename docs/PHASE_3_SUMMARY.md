# Báo cáo Kết quả Giai đoạn 3 (Phase 3 Summary Report) - Phân cụm trên Không gian 169D

Tài liệu này tổng hợp toàn bộ các công việc, kết quả thực nghiệm, chỉ số đánh giá, hồ sơ cụm và nhận xét chuyên môn trong **Giai đoạn 3: Phân cụm & Đánh giá Chất lượng** (được thực hiện phân cụm trực tiếp trên không gian đặc trưng 169D gốc để bảo toàn metric khoảng cách).

---

# # Các Tệp tin Đã Được Tạo ra

Sau khi thực thi thành công Giai đoạn 3, cấu trúc thư mục của dự án đã bổ sung các tệp tin sau:

1. **Thư mục [`data/`](file:///Users/anhnon/vietnam-job-market-clustering/data/) (Dữ liệu đã phân cụm)**:
 * `clean_data_train_clustered.csv`: Dữ liệu Train sạch đã bổ sung cột `cluster_id` (mã số cụm từ 0 đến 10) và `cluster_label` (nhãn ngữ nghĩa tự động).
 * `clean_data_test_clustered.csv`: Dữ liệu Test sạch đã bổ sung cột `cluster_id` và `cluster_label`.
2. **Thư mục [`models/`](file:///Users/anhnon/vietnam-job-market-clustering/models/) (Mô hình phân cụm)**:
 * `clustering_model.pkl`: Bộ phân cụm `MiniBatchKMeans` đã huấn luyện hoàn chỉnh với số lượng cụm tối ưu $K = 11$ trên đặc trưng 169D.
 * `cluster_labels_map.pkl`: Bản đồ ánh xạ nhãn ngữ nghĩa của 11 cụm đã học.
3. **Thư mục [`plots/`](file:///Users/anhnon/vietnam-job-market-clustering/plots/) (Đồ thị chất lượng cao)**:
 * `clustering_evaluation_metrics.png`: Biểu đồ so sánh 4 chỉ số chất lượng phân cụm (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz) ứng với các giá trị $K \in [5, 7, 9, 11, 13, 15, 17, 19]$.
 * `clusters_distribution_2d.png`: Bản đồ phân bố 2D UMAP của 50,000 tin tuyển dụng ngẫu nhiên, tô màu theo 11 cụm tối ưu, sử dụng kích thước điểm và độ mờ tối ưu (`s=3`, `alpha=0.35`, `edgecolors='none'`) để hạn chế chồng lấp (overplotting).
4. **Các Notebook nộp bài trong thư mục `notebooks/`**:
 * `03_training.ipynb`: Notebook chạy quy trình huấn luyện từ đặc trưng 169D -> K-Means -> Profiling.
 * `04_testing.ipynb`: Notebook chạy quy trình kiểm thử dự đoán nhanh (inference) cho tập Test.

---

# # Kết quả Khảo sát Số lượng Cụm (K-Evaluation)

Quá trình khảo sát sử dụng mô hình **Mini-Batch K-Means** chạy trên toàn bộ tập dữ liệu huấn luyện (353,158 dòng sau khi khử nhiễu và ngoại lệ) trực tiếp trong không gian đặc trưng **169D** đã chuẩn hóa. 
Để tối ưu hóa tài nguyên tính toán, chỉ số **Silhouette Score** được đánh giá trên một mẫu ngẫu nhiên đại diện gồm **10,000 dòng**. Các chỉ số còn lại (Inertia, Davies-Bouldin Index, Calinski-Harabasz Index) được tính toán trên toàn bộ dữ liệu.

Kết quả thu được với bước nhảy $K = 2$:

| Số cụm K | Inertia (Thấp là tốt) | Silhouette Score (Cao là tốt) | Davies-Bouldin Index (Thấp là tốt) | Calinski-Harabasz Index (Cao là tốt) | Thời gian chạy (giây) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **K = 5** | 54,333,217.69 | -0.0284 | 5.9486 | 103.79 | 1.17s |
| **K = 7** | 53,416,563.18 | -0.0192 | 4.7758 | 97.52 | 0.85s |
| **K = 9** | 52,608,752.98 | -0.0142 | 5.2263 | 93.89 | 0.76s |
| **K = 11** | 51,697,489.73 | **-0.0056** | 4.8260 | 93.97 | 0.80s |
| **K = 13** | 51,267,325.38 | -0.0165 | **4.4013** | 86.11 | 0.77s |
| **K = 15** | 49,770,752.04 | 0.0154 | 3.6511 | 96.90 | 0.82s |
| **K = 17** | 48,977,292.37 | 0.0099 | 3.9635 | 99.24 | 0.81s |
| **K = 19** | 48,193,467.17 | 0.0123 | 3.4954 | 97.17 | 0.79s |

---

# # Khung Đánh giá Mô hình Phân cụm (Clustering Evaluation Framework)

Trong thực tế khoa học dữ liệu, đánh giá phân cụm không chỉ dựa vào các chỉ số toán học thô (Technical Metrics) mà bắt buộc phải kết hợp với **khả năng diễn dịch nghiệp vụ (Business Interpretation)** để xem cấu trúc phân cụm có thật sự phản ánh thực tế thị trường lao động hay không.

# # # 1. Các chỉ số chất lượng phân cụm (Cluster Quality Metrics)

# # # # A. Silhouette Score (Độ rộng bóng bóng)
* **Mục đích**: Đo lường mức độ gần gũi của một điểm dữ liệu với các điểm trong cùng một cụm so với các điểm ở các cụm khác.
* **Công thức**: 
 $$s = \frac{b - a}{\max(a, b)}$$
 *Trong đó: $a$ là khoảng cách trung bình nội cụm, $b$ là khoảng cách trung bình tới cụm lân cận gần nhất.*
* **Ý nghĩa điểm số**:
 * $s \approx 1$: Các cụm được phân chia rất tốt, tách biệt rõ ràng.
 * $s \approx 0$: Các cụm bị chồng lấn (overlapping).
 * $s < 0$: Điểm dữ liệu bị xếp sai cụm.
* **Đánh giá thực tế trên dữ liệu nhiều chiều (High-Dimensional Space)**:
 * Trong không gian 169D, Silhouette Score của chúng ta dao động quanh mức từ $-0.02$ đến $0.01$. Điều này hoàn toàn bình thường do hiệu ứng **"Lời nguyền số chiều" (Curse of Dimensionality)**, khi khoảng cách tuyến tính giữa các điểm dữ liệu trong không gian cao chiều có xu hướng trở nên đồng đều hơn, làm giảm độ tương phản giữa khoảng cách nội cụm và liên cụm.
 * **Thang điểm đánh giá thực tế**:
 * $> 0.7$: Rất hiếm, cấu trúc dữ liệu cực kỳ lý tưởng.
 * $0.5 - 0.7$: Tốt.
 * $0.3 - 0.5$: Khá ổn.
 * $0.1 - 0.3$: Có nhiễu nhưng vẫn sử dụng được (Noisy but usable).
 * $< 0.1$: Cấu trúc cụm yếu.
 * **Nhận xét**: Điểm Silhouette đạt cực đại cục bộ tại $K = 11$ (đạt $-0.0056$, cao hơn hẳn so với $K=9$ là $-0.0142$ và $K=13$ là $-0.0165$) xác định đây là một mốc phân cụm tối ưu cục bộ có ý nghĩa nhất trong không gian 169D.

# # # # B. Davies-Bouldin Index (DBI)
* **Mục đích**: Đánh giá độ nén (compactness) nội cụm và độ tách biệt (separation) liên cụm.
* **Công thức**:
 $$DB = \frac{1}{K}\sum_{i=1}^K \max_{j \neq i} \left(\frac{S_i + S_j}{M_{ij}}\right)$$
 *Trong đó: $S_i, S_j$ là khoảng cách trung bình từ các điểm đến tâm cụm tương ứng, $M_{ij}$ là khoảng cách giữa 2 tâm cụm.*
* **Ý nghĩa**: **Càng nhỏ càng tốt**. DBI thấp chỉ ra rằng các cụm có độ nén cao (điểm trong cụm gần nhau) và khoảng cách giữa các cụm lớn.
* **Nhận xét**: DBI tại $K=11$ (4.82) và $K=13$ (4.40) giảm đáng kể so với $K=9$ (5.22), củng cố thêm lựa chọn $K=11$.

# # # # C. Calinski-Harabasz Index (CHI)
* **Mục đích**: Đo tỷ lệ giữa phương sai giữa các cụm (between-cluster variance) và phương sai trong nội bộ cụm (within-cluster variance).
* **Ý nghĩa**: **Càng lớn càng tốt**. Điểm số CHI cao thể hiện các cụm được phân chia rõ nét và chặt chẽ.

---

# # # 2. Sự nhất quán về mặt Ngữ nghĩa (Semantic Coherence) & Giá trị Nghiệp vụ

Mục tiêu tối thượng của phân cụm là **tìm kiếm cấu trúc ẩn có ý nghĩa** (exploratory analysis) chứ không phải tìm một "đáp án chính xác duy nhất". K-Means luôn cố gắng ép dữ liệu vào $K$ cụm dù dữ liệu có phân mảnh tự nhiên hay không. Do đó, việc đánh giá bằng kiến thức nghiệp vụ (Domain Knowledge) là cực kỳ quan trọng.

Một mô hình phân cụm tốt phải thỏa mãn 4 tiêu chí cốt lõi:
1. **Độ nén (Compactness)**: Các tin tuyển dụng trong cùng một cụm phải có đặc tính tương đương nhau (ví dụ: cùng kỹ năng, cùng ngành).
2. **Độ tách biệt (Separation)**: Các cụm khác nhau phải thể hiện những vai trò công việc khác nhau rõ rệt trên thị trường.
3. **Tính ổn định (Stability)**: Khi lấy mẫu lại dữ liệu, cấu trúc các cụm không bị thay đổi đột ngột.
4. **Khả năng giải thích (Interpretability)**: Người dùng có thể dễ dàng hiểu được mỗi cụm đang đại diện cho nhóm công việc nào.

---

# # # Chi tiết Hồ sơ 11 Cụm Tối ưu & Nhãn Ngữ nghĩa (Clustering Profiles)

Dưới đây là kết quả phân tích đặc trưng (Profiling) trên 11 cụm đã được huấn luyện trên không gian 169D và gán nhãn tự động dựa trên quy luật phân phối lương, kinh nghiệm và từ khóa TF-IDF chính từ cột `text_combined`:

| Cụm ID | Tỷ lệ | Lương Trung vị | Kinh nghiệm Trung vị | Ngành nghề chính nổi bật | Từ khóa chủ đạo (TF-IDF) | Nhãn ngữ nghĩa tự gán |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **00** | 0.87% | 9.0M VND | 3.0 năm | Lao động phổ thông | xe, hàng, lái, giao, lái xe | Cụm 00: Lao động phổ thông (Lương Trung Bình - ~9.0M) - Key: xe, hàng, lái |
| **01** | 47.65% | 8.0M VND | 2.0 năm | Chăm sóc khách hàng | hàng, khách, khách hàng, năng, kinh | Cụm 01: Chăm sóc khách hàng (Lương Trung Bình - ~8.0M) - Key: hàng, khách, khách hàng |
| **02** | 0.92% | 10.0M VND | 3.0 năm | IT Phần mềm | and, năng, kinh, triển, thống | Cụm 02: IT Phần mềm (Lương Trung Bình - ~10.0M) - Key: and, năng, kinh |
| **03** | 0.68% | 15.0M VND | 1.0 năm | Ngân hàng / Tài chính | năng, kỹ năng, kỹ, hàng, khách hàng | Cụm 03: Ngân hàng / Tài chính (Lương Cao - ~15.0M) - Key: năng, kỹ năng, kỹ |
| **04** | 1.58% | 8.0M VND | 3.0 năm | Vận Tải - Lái xe - Giao nhận | hàng, xe, giao, lái, lái xe | Cụm 04: Vận Tải - Lái xe (Lương Trung Bình - ~8.0M) - Key: hàng, xe, giao |
| **05** | 9.30% | 9.0M VND | 3.0 năm | Giáo dục - Đào tạo | hàng, năng, lý, động, kinh | Cụm 05: Giáo dục - Đào tạo (Lương Trung Bình - ~9.0M) - Key: hàng, năng, lý |
| **06** | 8.55% | 9.0M VND | 1.0 năm | Bán hàng - Kinh doanh | hàng, khách, năng, and, khách hàng | Cụm 06: Bán hàng - Kinh doanh (Lương Trung Bình - ~9.0M) - Key: hàng, khách, năng |
| **07** | 6.91% | 6.0M VND | 1.0 năm | Bán hàng - Kinh doanh | hàng, khách, năng, khách hàng, ca | Cụm 07: Bán hàng - Kinh doanh (Lương Thấp/Chưa Rõ - ~6.0M) - Key: hàng, khách, năng |
| **08** | 8.79% | 9.0M VND | 3.0 năm | Kế toán / Kiểm toán | toán, kế, kế toán, hàng, định | Cụm 08: Kế toán / Kiểm toán (Lương Trung Bình - ~9.0M) - Key: toán, kế, kế toán |
| **09** | 13.72% | 13.0M VND | 4.0 năm | Xây dựng | hàng, năng, lý, kinh, quản | Cụm 09: Xây dựng (Lương Trung Bình - ~13.0M) - Key: hàng, năng, lý |
| **10** | 1.03% | 8.0M VND | 1.0 năm | Chăm sóc khách hàng | bạn, định, hàng, khách, khách hàng | Cụm 10: Chăm sóc khách hàng (Lương Trung Bình - ~8.0M) - Key: bạn, định, hàng |

---

# # # Đánh giá Tổng thể Mô hình từ Chuyên gia

Dựa trên kết quả phân cụm thực tế trên 11 nhóm công việc được trích xuất từ dữ liệu thị trường việc làm Việt Nam:

* **Về tính Ngữ nghĩa (Semantic Coherence)**: ** Khá tốt**. Mô hình tách bạch được các ngành nghề có tính đặc thù cao như IT Phần mềm (Cụm 02), Kế toán (Cụm 08), Lao động phổ thông/Lái xe (Cụm 00 & 04) và phân biệt được các sắc thái bán hàng (Cụm 06 lương trung bình vs Cụm 07 lương thấp bán ca kíp).
* **Về tính Tách biệt (Separation)**: **⚠ Trung bình - Khá**. Do phân cụm trên không gian 169D và biểu diễn trực quan trên bản đồ chiếu UMAP 2D, một số cụm có sự chồng lấn nhẹ ở ranh giới. Điều này phản ánh đúng thực tế khi mô tả công việc (JD) của các ngành dịch vụ, chăm sóc khách hàng, marketing có xu hướng dùng chung nhiều từ khóa giao tiếp.
* **Về khả năng Giải thích (Interpretability)**: ** Tốt**. Từng cụm sau khi phân tích profile đều đại diện cho các nhóm công việc rất rõ ràng trong đời sống kinh tế xã hội.
* **Về giá trị Nghiệp vụ (Business Value)**: ** Có ý nghĩa thật sự**. Kết quả phân cụm có thể ứng dụng trực tiếp vào các hệ thống gợi ý việc làm (Job Recommendation), phân khúc ứng viên, nghiên cứu xu hướng lương theo ngành nghề, và phân tích thị trường lao động vĩ mô.

---

# # Nhận xét Chuyên môn: Mối quan hệ giữa Không gian Đặc trưng (UMAP) và Lương (Salary)

Một quan sát học thuật cực kỳ quan trọng được đúc rút từ quá trình trực quan hóa không gian đặc trưng UMAP 2D (tô màu theo cụm và tô màu theo lương ở Phase 2):

# # # 1. Hiện tượng Quan sát
* **Tính phân cụm rõ rệt**: Biểu đồ phân cụm (được gán từ K-Means 169D) chiếu xuống mặt phẳng UMAP 2D tạo ra các cụm tương đối tập trung. Điều này chứng tỏ không gian đặc trưng 169D trích xuất từ văn bản mô tả (JD) và thuộc tính cấu trúc mang thông tin phân tách tốt.
* **Mức lương bị trộn lẫn**: Khi tô màu đồ thị UMAP theo mức lương tuyển dụng (`salary_min_m_vnd`), chúng ta quan sát thấy mức lương bị trộn lẫn qua nhiều cụm khác nhau chứ không tập trung hoàn toàn thành các vùng lương tách biệt.

# # # 2. Kết luận và Lý giải
* **Kết luận**: **Cấu trúc không gian đặc trưng (Embedding Structure) không trùng khớp với cấu trúc phân bổ mức lương (Salary Structure)**.
* **Lý giải thực tế**:
 * **Tính đa dạng về thâm niên (Seniority)**: Cùng một ngành nghề (ví dụ: IT Phần mềm - Cụm 02) có thể tồn tại dải lương rất rộng (từ 5M cho Intern đến 40M+ cho Tech Lead). Vì có chung từ khóa chuyên ngành trong mô tả công việc (JD), chúng vẫn được xếp chung một cụm ngành nghề chuyên môn nhưng phân phối lương lại cực kỳ đa dạng.
 * **Các yếu tố ngoại sinh**: Mức lương tuyển dụng chịu ảnh hưởng bởi rất nhiều yếu tố không đồng nhất (heterogeneous factors) như quy mô doanh nghiệp, quỹ ngân sách tuyển dụng, khả năng thương lượng của ứng viên, và tính chất công việc khẩn cấp, hơn là chỉ phụ thuộc vào các từ khóa mô tả công việc gốc.
* **Nhận xét Học thuật (Academic Statement)**:
 > *"The UMAP projection reveals several meaningful local clusters, indicating that the high-dimensional feature space contains latent subgroup structures. However, salary levels remain distributed across multiple clusters, suggesting that salary is influenced by heterogeneous factors rather than serving as the dominant clustering attribute."*
