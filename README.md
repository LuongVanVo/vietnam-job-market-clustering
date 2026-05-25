# Vietnam Job Market Clustering - Phases 1, 2, & 3

Dự án này thực hiện phân cụm và phân tích cấu trúc bộ dữ liệu tuyển dụng Việt Nam (`tinixai/vietnamese-job-descriptions` từ Hugging Face) gồm 3 giai đoạn:
1.  **Giai đoạn 1 (Phase 1)**: Tiền xử lý, trích xuất thuộc tính số (Lương, Kinh nghiệm), cap ngoại lệ và làm sạch văn bản.
2.  **Giai đoạn 2 (Phase 2)**: Mã hóa đặc trưng, vector hóa TF-IDF + SVD mô tả công việc, lọc ngoại lệ không gian vector (Outlier Tầng 2 bằng Isolation Forest), và giảm chiều bằng UMAP.
3.  **Giai đoạn 3 (Phase 3)**: Phân cụm Mini-Batch K-Means, khảo sát số cụm tối ưu, gán nhãn cụm tự động bằng từ khóa và phân tích phân phối thuộc tính, trực quan hóa ranh giới cụm.

---

## 📂 Cấu trúc Thư mục & Tệp tin Chính

*   `preprocess.py`: Script chạy pipeline làm sạch dữ liệu và tách tập train/test (Giai đoạn 1).
*   `feature_engineering.py`: Script chạy pipeline mã hóa đặc trưng, vector hóa text, lọc ngoại lệ Isolation Forest và giảm chiều UMAP (Giai đoạn 2).
*   `clustering.py`: Script chạy pipeline phân cụm Mini-Batch K-Means, tính toán các chỉ số đánh giá, gán nhãn cụm và trực quan hóa (Giai đoạn 3).
*   `notebooks/`: Thư mục chứa các Jupyter Notebook chạy tương tác cho từng giai đoạn:
    *   `preprocess.ipynb`: Notebook Giai đoạn 1.
    *   `feature_engineering.ipynb`: Notebook Giai đoạn 2.
    *   `clustering.ipynb`: Notebook Giai đoạn 3.
*   `data/`: Thư mục lưu trữ dữ liệu (đã bỏ qua Git do dung lượng lớn):
    *   `raw_data_train.csv` & `raw_data_test.csv`: Dữ liệu thô phân chia 90/10.
    *   `clean_data_train.csv` & `clean_data_test.csv`: Dữ liệu sạch sau Giai đoạn 1.
    *   `clean_data_train_final.csv`: Dữ liệu Train sạch sau khi lọc ngoại lệ Giai đoạn 2.
    *   `features_train.npz` & `features_test.npz`: Ma trận đặc trưng UMAP 20D (clustering) và 2D (visualization).
    *   `clean_data_train_clustered.csv` & `clean_data_test_clustered.csv`: Dữ liệu đầu ra cuối cùng đã gán nhãn cụm.
*   `models/`: Lưu trữ các mô hình tiền xử lý và phân cụm đã fit (`.pkl`).
*   `plots/`: Lưu trữ các biểu đồ phân tích và trực quan hóa kết quả.

---

## 🛠️ Hướng dẫn Chạy Chương trình

### 1. Cài đặt các thư viện cần thiết
Sử dụng môi trường ảo Python thích hợp (ví dụ: `/Users/anhnon/my_virtualenvs/khdl`) và cài đặt các thư viện:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn datasets umap-learn joblib ipykernel
```

### 2. Thực thi Giai đoạn 1 (Preprocessing & Cleaning)
Chạy script tự động trên toàn bộ dữ liệu:
```bash
python preprocess.py
```
*(Hỗ trợ tham số `--sample 10000` nếu muốn chạy thử nhanh trên mẫu nhỏ).*

### 3. Thực thi Giai đoạn 2 (Feature Engineering & UMAP)
Chạy mã hóa đặc trưng và giảm chiều:
```bash
python feature_engineering.py
```

### 4. Thực thi Giai đoạn 3 (Clustering & Evaluation)
Chạy phân cụm, khảo sát số cụm tối ưu và gán nhãn ngữ nghĩa:
```bash
python clustering.py
```
*   Mặc định sẽ khảo sát $K \in [5, 7, 9, 11, 13, 15, 17, 19]$ với bước nhảy là 2.
*   Bạn có thể điều chỉnh số lượng cụm tối ưu mong muốn bằng tham số `--k_optimal` (mặc định $K^*=10$):
    ```bash
    python clustering.py --k_optimal 12
    ```
*   Có thể bỏ qua bước khảo sát chỉ số để chạy nhanh hơn bằng tham số `--skip_eval`:
    ```bash
    python clustering.py --skip_eval
    ```

---

## 🧼 Chi tiết Kỹ thuật Phân cụm & Nhận xét Chuyên môn

### 1. Khảo sát chất lượng phân cụm
Hệ thống sử dụng các chỉ số **Inertia**, **Silhouette Score** (tính trên mẫu 10k), **Davies-Bouldin Index (DBI)**, và **Calinski-Harabasz Index (CHI)** để đánh giá các cụm. Số cụm $K = 10$ được lựa chọn làm điểm tối ưu nhờ cân bằng tốt giữa độ chặt chẽ cấu trúc hình học và tính thực tế khi phân tách thị trường.

### 2. Nhãn ngữ nghĩa tự gán của 10 cụm tối ưu
Mỗi cụm được tự động phân tích và gán nhãn dựa trên tổ hợp ngành nghề phổ biến, mức lương trung vị, kinh nghiệm trung vị, và top từ khóa đặc trưng nhất từ mô tả công việc (TF-IDF):
*   **Cụm 00**: Bán hàng (Lương Trung Bình - ~8.0M) - Key: hàng, khách, khách hàng
*   **Cụm 01**: Xây dựng (Lương Trung Bình - ~10.0M) - Key: năng, hàng, kỹ
*   **Cụm 02**: Kế toán (Lương Trung Bình - ~9.0M) - Key: toán, kế, kế toán
*   **Cụm 03**: Xuất Nhập Khẩu (Lương Trung Bình - ~10.0M) - Key: hàng, năng, khách
*   ...

### 3. Nhận xét chuyên môn (Academic Observation)
> *"The UMAP projection reveals several meaningful local clusters, indicating that the high-dimensional feature space contains latent subgroup structures. However, salary levels remain distributed across multiple clusters, suggesting that salary is influenced by heterogeneous factors rather than serving as the dominant clustering attribute."*
>
> *(Phép chiếu UMAP cho thấy sự tồn tại của nhiều cụm cục bộ có ý nghĩa, chứng tỏ rằng không gian đặc trưng nhiều chiều chứa các cấu trúc nhóm ẩn. Tuy nhiên, các mức lương vẫn phân tán trên nhiều cụm khác nhau, cho thấy lương chịu ảnh hưởng bởi các yếu tố không đồng nhất như kinh nghiệm, địa điểm, quy mô công ty, thâm niên và kỹ năng đặc thù hơn là đóng vai trò là đặc trưng phân cụm chủ đạo).*
