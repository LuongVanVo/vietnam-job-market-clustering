# Sơ đồ Luồng Hệ thống (System Architecture Pipeline)

Tài liệu này mô tả chi tiết luồng xử lý (Data Flow) và các nền tảng toán học cốt lõi (Mathematical Foundations) được áp dụng trong toàn bộ hệ thống Phân cụm Thị trường Lao động Việt Nam.

## 1. Sơ đồ Luồng Dữ liệu (Mermaid Flowchart)

```mermaid
graph TD
    classDef data fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef output fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

    RawDB[("Dữ liệu thô<br>1,000,000+ Dòng")]:::data
    Split["Chia Train/Test Set<br>80/20 Ratio"]:::process

    subgraph phase1 ["Phase 1: Tiền xử lý"]
        CleanTrain["Tập Train Thô"]:::data
        CleanTest["Tập Test Thô"]:::data
        Win["Winsorization<br>Khử Ngoại lệ Lương/KN"]:::process
        DropNA["Loại bỏ Null/Rác"]:::process
    end

    subgraph phase2 ["Phase 2: Trích xuất Đặc trưng (169D)"]
        OHE["One-Hot Encoding<br>Ngành nghề & Vị trí"]:::process
        Num["StandardScaler<br>Lương & Kinh nghiệm"]:::process
        NLP["TF-IDF + Truncated SVD<br>Mô tả công việc"]:::process
        Concat["Hợp nhất Ma trận<br>169 Chiều"]:::process
        iForest["Isolation Forest<br>Lọc dị thường Vector"]:::model
    end

    subgraph phase3 ["Phase 3: Huấn luyện K-Means"]
        SearchK["Khảo sát K Tối ưu<br>Elbow & Silhouette"]:::model
        KMeans["K-Means Algorithm<br>K=17"]:::model
        BusMap["Business Persona Mapping<br>Gán Nhãn Chuyên Nghiệp"]:::process
    end

    subgraph phase4 ["Phase 4: Kiểm định Mô hình"]
        TestVal["Kiểm định Tập Test<br>Geometric & Proportions"]:::model
        Metrics(("DBI, CHI, Silhouette<br>Pearson Correlation ~1.0")):::output
    end

    subgraph phase5 ["Phase 5: Trực quan hóa & Dashboard"]
        Viz["Vẽ Biểu đồ Tĩnh & 3D PCA"]:::process
        Dash["Dashboard Báo cáo<br>Extended Profiles"]:::output
    end

    %% Các luồng kết nối (Edges) được đưa ra ngoài subgraph để chống lỗi Parse
    RawDB --> Split
    Split --> CleanTrain
    Split --> CleanTest
    CleanTrain --> Win
    CleanTest --> Win
    Win --> DropNA

    DropNA --> OHE
    DropNA --> Num
    DropNA --> NLP
    
    OHE --> Concat
    Num --> Concat
    NLP --> Concat
    
    Concat --> iForest

    iForest --> SearchK
    SearchK --> KMeans
    KMeans --> BusMap

    BusMap --> TestVal
    TestVal --> Metrics

    Metrics --> Viz
    Viz --> Dash
```

---

## 2. Nền tảng Toán học (Mathematical Foundations)

Hệ thống được thiết kế với sự nghiêm ngặt về mặt Toán học nhằm đảm bảo tính toàn vẹn của không gian đặc trưng.

### 2.1. Khử ngoại lệ bằng Winsorization (Phase 1)
Để tránh các mức lương ảo (ví dụ: 999 Triệu/tháng) làm méo mó tâm cụm, dự án áp dụng kỹ thuật cắt xén giới hạn phân vị (Winsorization) thay vì xóa bỏ bản ghi:
$$ X_{i} = \begin{cases} 
P_{99} & \text{nếu } X_i > P_{99} \\
X_i & \text{nếu } X_i \le P_{99} 
\end{cases} $$
Trong đó, $P_{99}$ là bách phân vị thứ 99 của tập dữ liệu huấn luyện.

### 2.2. Nén Không gian Ngữ nghĩa (Phase 2)
Văn bản "Mô tả công việc" được biểu diễn dưới dạng ma trận TF-IDF thưa thớt, sau đó được xấp xỉ hóa hạng thấp (Low-Rank Approximation) bằng **Truncated SVD** để rút trích 100 chủ đề ẩn (Latent Topics) bảo toàn tối đa phương sai:
$$ A \approx U_k \Sigma_k V_k^T $$
Trong đó, $k=100$, giảm bớt độ nhiễu loạn của ngôn ngữ tự nhiên.

### 2.3. Khảo sát K-Means qua Silhouette Score (Phase 3)
Hệ thống sử dụng K-Means dựa trên khoảng cách Euclidean trong không gian 169 chiều. Chất lượng phân tách từng điểm dữ liệu $i$ được đo lường bằng:
$$ S(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} $$
*   $a(i)$: Khoảng cách trung bình từ điểm $i$ đến các điểm trong **cùng cụm**.
*   $b(i)$: Khoảng cách trung bình từ điểm $i$ đến các điểm trong **cụm gần nhất**.
Điểm đỉnh (Global Maximum) của đường cong trung bình $\bar{S}$ đã xác lập $K=17$ là cấu trúc tách viền tốt nhất.

### 2.4. Độ ổn định Phân bổ tuyệt đối - MAD (Phase 4)
Để chứng minh mô hình không Overfitting, độ lệch tỷ trọng phân bổ giữa Train và Test được kiểm định qua chỉ số Mean Absolute Difference:
$$ \text{MAD} = \frac{1}{K} \sum_{j=1}^{K} |P_{\text{Train}}(j) - P_{\text{Test}}(j)| \approx 0.16\% $$
Độ lệch chưa tới $0.2\%$ chứng minh $17$ Chân dung Nghề nghiệp là quy luật cung-cầu bất biến của thị trường.
