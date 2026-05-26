# BÁO CÁO GIAI ĐOẠN 3: HUẤN LUYỆN MÔ HÌNH PHÂN CỤM (TRAINING PIPELINE)
**(Unsupervised Clustering with MiniBatch K-Means)**

Tài liệu này tổng hợp quá trình huấn luyện mô hình học máy không giám sát (Unsupervised Learning) nhằm phân nhóm hơn 520,000 tin tuyển dụng trên thị trường lao động Việt Nam.

---

## 1. Kiến trúc Huấn luyện (Training Architecture)

Toàn bộ quy trình diễn ra tại file `notebooks/03_training.ipynb`:
- **Dữ liệu đầu vào:** Ma trận đặc trưng **169D** (169 chiều) đã được chuẩn hoá `StandardScaler` từ Phase 2. Mọi khoảng cách (Euclid) giữa biến văn bản (100D), biến phân loại (64D) và biến số (5D) đều được đưa về trọng số công bằng.
- **Thuật toán cốt lõi:** Sử dụng `MiniBatchKMeans(batch_size=2048)`. Do tập dữ liệu khổng lồ, thuật toán K-Means truyền thống (tính khoảng cách toàn cục) sẽ làm tràn bộ nhớ (OOM). Biến thể Mini-Batch lấy mẫu ngẫu nhiên giúp hội tụ cực nhanh (vài giây) mà vẫn đảm bảo độ chính xác.

---

## 2. Quá trình "Cân Não" chọn K (Hyperparameter Tuning)

Quá trình quét mốc $K \in [5, 19]$ trên mẫu 10,000 dòng đã phơi bày một cuộc "xung đột chỉ số" kinh điển trong các bài toán NLP (Xử lý ngôn ngữ tự nhiên):
- **Davies-Bouldin Index (DBI):** Đạt điểm đẹp nhất ở $K = 15$.
- **Calinski-Harabasz (CHI):** Tạo đỉnh ở $K = 7$ và $K = 11$.
- **Silhouette Score:** Tạo một đỉnh chóp (Global Maximum) khổng lồ tại **$K = 17$** (đạt $0.041$, cao gấp đôi mức trung bình).

> [!TIP]
> **Quyết định thiết kế:** Thay vì chọn $K=11$ (trung hoà), dự án quyết định chốt **$K = 17$**. Trong không gian 169D siêu thưa thớt, chỉ số **Silhouette** là "kim chỉ nam" đáng tin cậy nhất để đo lường việc các điểm dữ liệu nằm khít trong cụm của nó và tách biệt khỏi cụm hàng xóm. Số lượng 17 cụm cũng bám sát thực tiễn phân hóa phức tạp của thị trường lao động.

---

## 3. Khai phá Tri thức (Knowledge Discovery) từ 17 Cụm

Với $K=17$, kết hợp cùng kỹ thuật trích xuất từ khóa TF-IDF nội bộ từng cụm (`Dynamic Semantic Labeling`), mô hình đã gặt hái thành công vang dội khi không chỉ gom được các khối ngành lớn mà còn "bóc tách" (isolate) được các ngành ngách đắt giá:

### Nhóm Chuyên môn hẹp - Thành tựu lớn nhất của mốc 17
1. **Cụm 09 (Tài chính - Ngân hàng):** Dù chỉ chiếm $0.69\%$ thị trường và yêu cầu kinh nghiệm rất thấp (1 năm), nhưng mức lương Median chạm đỉnh toàn bảng: **15.0 Triệu VND**. (Từ khoá: `thanh toán`).
2. **Cụm 02 (Công nghệ thông tin - IT):** Nằm tại thị trường Hà Nội, mức lương $10.0M$ dù chỉ 2 năm kinh nghiệm. (Từ khoá: `phát triển`, `kỹ năng`).
3. **Cụm 15 (Quản lý chất lượng - QA/QC):** Lương $10.0M$. (Từ khóa cực nét: `chất lượng`, `sản phẩm`).
4. **Cụm 00 (Thiết kế - Sáng tạo):** Lương $10.0M$. (Từ khoá: `thiết kế`, `sản phẩm`).

### Nhóm Ngành công nghiệp Vĩ mô
- **Cụm 13 (Kỹ sư Xây dựng):** Quy mô $6.50\%$, lương $10.0M$, kinh nghiệm $3.0$ năm. (Từ khóa: `thi công`, `công trình`, `thiết kế`).
- **Cụm 12 (Cơ khí & Sản xuất):** Quy mô $6.99\%$, lương $10.0M$. (Từ khóa: `điện`, `máy móc`).
- **Cụm 10 (Kế toán / Kiểm toán):** Trụ cột tài chính doanh nghiệp, chiếm $11.13\%$. (Từ khóa: `kế toán`, `định khoản`).

### Nhóm Dịch vụ & Lao động phổ thông (Lực lượng đông đảo)
- **Cụm 01 (CSKH / Telesales):** Chiếm tới $22.58\%$, lương $8.0M$.
- **Cụm 11 (LĐPT / Vận tải / Bảo vệ):** Chiếm $10.42\%$, lương $8.0M$. (Từ khóa: `xe`, `bảo vệ`).
- **Cụm 06 (Thực tập sinh):** Lương hỗ trợ $4.0M$, yêu cầu kinh nghiệm bằng $0$ hoặc dưới $1$ năm.

---

## 4. Dữ liệu Kết Xuất (Output Artifacts)

Mô hình đã ghi nhận toàn bộ quá trình và tự động dán nhãn lại cho $>520,000$ tin tuyển dụng ban đầu:
- **`models/clustering_model.pkl`**: Trọng lượng mô hình (Centroids) của 17 cụm, dùng để dự đoán real-time cho tin tuyển dụng mới ở Giai đoạn 4.
- **`models/cluster_labels_map.pkl`**: Từ điển ánh xạ từ `Cluster ID (0-16)` sang tên gọi ngữ nghĩa của ngành (được sinh tự động).
- **`results/clean_data_train_clustered.csv`**: Bộ dữ liệu huấn luyện đã được dính kèm 2 cột quý giá: `cluster_id` và `cluster_label`, sẵn sàng cho công tác dựng Dashboard (BI) báo cáo.
