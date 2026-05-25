import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

# Ordinal mapping for Education Level
EDUCATION_MAP = {
    'Không': 0, 'Trung học': 1, 'Chứng chỉ': 2, 'Trung cấp': 3, 
    'Bằng cấp liên quan': 3, 'Cao đẳng': 4, 'Đại học': 5, 
    'Cử nhân': 5, 'Kỹ sư': 5, 'Khác': 0
}

def process_pipeline():
    print("==================================================")
    print("PHASE 2: FEATURE ENGINEERING & DIMENSION REDUCTION")
    print("==================================================")
    
    t_start = time.time()
    
    # Create models folder
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Load cleaned datasets
    print("\n[Step 1] Loading cleaned datasets...")
    train_path = "data/clean_data_train.csv"
    test_path = "data/clean_data_test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Cleaned files not found. Please run Phase 1 preprocessing first.")
        return
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Loaded Train: {df_train.shape} | Test: {df_test.shape}")
    
    # 2. Encode structured fields
    print("\n[Step 2] Encoding structured fields...")
    
    # A. Ordinal Encoding for Education Level
    print(" - Ordinal encoding for 'education_level'...")
    df_train['edu_encoded'] = df_train['education_level'].map(EDUCATION_MAP).fillna(0).astype(float)
    df_test['edu_encoded'] = df_test['education_level'].map(EDUCATION_MAP).fillna(0).astype(float)
    
    # B. Scaling Salary and Experience Level
    print(" - Scaling numerical features (Salary, Experience)...")
    num_cols = ['salary_min_m_vnd', 'salary_max_m_vnd', 'exp_min_years', 'exp_max_years', 'edu_encoded']
    scaler_num = MinMaxScaler()
    
    num_train = scaler_num.fit_transform(df_train[num_cols])
    num_test = scaler_num.transform(df_test[num_cols])
    
    # C. One-Hot Encoding for categorical fields
    print(" - One-Hot encoding for categorical fields (location, job_type, job_industry, job_position)...")
    cat_cols = ['location', 'job_type', 'job_industry', 'job_position']
    
    # Use min_frequency=0.005 (groups rare categories appearing < 0.5% into infrequent category)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=0.005)
    
    cat_train = ohe.fit_transform(df_train[cat_cols])
    cat_test = ohe.transform(df_test[cat_cols])
    
    print(f"One-Hot features dimension: {cat_train.shape[1]} dimensions (derived from unique values)")
    
    # Combine structured features
    struct_train = np.hstack([num_train, cat_train])
    struct_test = np.hstack([num_test, cat_test])
    print(f"Total Structured features shape: Train: {struct_train.shape} | Test: {struct_test.shape}")
    
    # 3. Text Vectorization using TF-IDF + TruncatedSVD
    print("\n[Step 3] Vectorizing text combined fields using TF-IDF + SVD...")
    
    # TF-IDF Configuration
    # Vietnamese stop words list
    VIETNAMESE_STOP_WORDS = [
        'và', 'của', 'để', 'cho', 'có', 'trong', 'một', 'là',
        'các', 'được', 'với', 'những', 'tại', 'này', 'theo', 'về',
        'ra', 'đã', 'sẽ', 'như', 'khi', 'lên', 'từ', 'nhiều',
        'vào', 'hoặc', 'nếu', 'lại', 'đang', 'cùng', 'qua', 'trước',
        'sau', 'khoảng', 'trên', 'dưới', 'công', 'ty', 'tuyển', 'dụng',
        'yêu', 'cầu', 'làm', 'việc', 'nhân', 'viên', 'vị', 'trí',
        'chúng', 'tôi', 'quyền', 'lợi', 'chế', 'độ', 'hồ', 'sơ',
        'nộp', 'liên', 'hệ', 'tin', 'tức', 'thông', 'báo', 'mức',
        'lương', 'yêu cầu', 'làm việc', 'nhân viên', 'công ty', 'hồ sơ', 'liên hệ', 'quyền lợi',
        'chế độ', 'tuyển dụng', 'đáp', 'ứng', 'công việc', 'công tác', 'thực hiện', 'tham gia',
        'hỗ trợ', 'phát triển', 'yêu cầu công việc', 'báo cáo', 'quản lý', 'kỹ năng', 'khả năng', 'kinh nghiệm',
        'tốt nghiệp', 'chuyên ngành', 'phù hợp', 'có thể', 'được hưởng', 'được đóng', 'được đào tạo', 'chuyên',
        'cáo', 'gia', 'hiện', 'hưởng', 'hỗ', 'hợp', 'khả', 'kinh',
        'kỹ', 'lý', 'nghiệm', 'nghiệp', 'ngành', 'năng', 'phát', 'phù',
        'quản', 'tham', 'thể', 'thực', 'triển', 'trợ', 'tác', 'tạo',
        'tốt', 'đào', 'đóng'
    ]

    tfidf = TfidfVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words=VIETNAMESE_STOP_WORDS
    )
    
    print(" - Fitting TF-IDF Vectorizer on train combined text...")
    t0 = time.time()
    tfidf_train = tfidf.fit_transform(df_train['text_combined'].fillna(""))
    tfidf_test = tfidf.transform(df_test['text_combined'].fillna(""))
    print(f" - TF-IDF matrix generated in {time.time() - t0:.2f}s. Shape: {tfidf_train.shape}")
    
    # Reduce dimension using TruncatedSVD (to 100 components)
    n_components = 100
    print(f" - Reducing dimension to {n_components} components using TruncatedSVD...")
    t0 = time.time()
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    text_train = svd.fit_transform(tfidf_train)
    text_test = svd.transform(tfidf_test)
    print(f" - SVD reduction finished in {time.time() - t0:.2f}s. Shape: {text_train.shape}")
    
    # 4. Combine text and structured features
    print("\n[Step 4] Stacking structured and text feature matrices...")
    full_train = np.hstack([struct_train, text_train])
    full_test = np.hstack([struct_test, text_test])
    print(f"Combined feature shape: Train: {full_train.shape} | Test: {full_test.shape}")
    
    # 5. Outlier Detection (Tầng 2 - Vector Space Outliers)
    print("\n[Step 5] Detecting vector-space outliers using Isolation Forest on Train set...")
    t0 = time.time()
    # contamination=0.04 (approx 4% expected outlier in job listings)
    iso_forest = IsolationForest(contamination=0.04, n_estimators=100, random_state=42, n_jobs=-1)
    
    outlier_labels = iso_forest.fit_predict(full_train)
    inliers_mask = outlier_labels == 1
    
    # Filter training set
    df_train_clean = df_train[inliers_mask].copy()
    full_train_clean = full_train[inliers_mask]
    
    print(f" - Isolation Forest finished in {time.time() - t0:.2f}s.")
    print(f" - Removed {np.sum(~inliers_mask)} outliers ({np.mean(~inliers_mask)*100:.2f}%).")
    print(f" - Cleaned Train Feature shape: {full_train_clean.shape}")
    
    # Save clean dataset after Phase 2 outliers filtering
    df_train_clean.to_csv("results/clean_data_train_final.csv", index=False)
    
    # 6. Scaling final combined matrix
    print("\n[Step 6] Normalizing combined feature matrix...")
    scaler_final = StandardScaler()
    full_train_scaled = scaler_final.fit_transform(full_train_clean)
    full_test_scaled = scaler_final.transform(full_test)
    
    # 7. Dimensionality Reduction using UMAP
    print("\n[Step 7] Running UMAP dimensionality reduction...")
    # Because UMAP on 500k rows is slow and memory-intensive, we sample 50,000 rows to fit the UMAP model,
    # and then project (transform) all training and testing rows.
    sample_size = min(50000, len(full_train_scaled))
    print(f" - Sampling {sample_size} rows to fit UMAP models...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(full_train_scaled), sample_size, replace=False)
    sample_data = full_train_scaled[sample_indices]
    
    # UMAP for Clustering (20 dimensions)
    print(" - Fitting UMAP for Clustering (20 components, cosine metric)...")
    t0 = time.time()
    reducer_cluster = umap.UMAP(
        n_components=20,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        low_memory=True
    )
    reducer_cluster.fit(sample_data)
    print(f"   UMAP cluster model fitted in {time.time() - t0:.2f}s.")
    
    # UMAP for Visualization (2 dimensions)
    print(" - Fitting UMAP for Visualization (2 components, cosine metric)...")
    t0 = time.time()
    reducer_viz = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric='cosine',
        random_state=42,
        low_memory=True
    )
    reducer_viz.fit(sample_data)
    print(f"   UMAP visualization model fitted in {time.time() - t0:.2f}s.")
    
    # Transform full datasets
    print(" - Projecting full Train and Test datasets...")
    t0 = time.time()
    train_20d = reducer_cluster.transform(full_train_scaled)
    test_20d = reducer_cluster.transform(full_test_scaled)
    
    train_2d = reducer_viz.transform(full_train_scaled)
    test_2d = reducer_viz.transform(full_test_scaled)
    print(f"   Projections completed in {time.time() - t0:.2f}s.")
    
    # 8. Save Feature matrices and Pipeline Models
    print("\n[Step 8] Saving feature matrices and fitted pipeline models...")
    np.savez("results/features_train.npz", features_160d=full_train_scaled)
    np.savez("results/features_test.npz", features_160d=full_test_scaled)
    print(" - Saved features matrices to 'results/features_train.npz' and 'results/features_test.npz'.")
    
    # Save model pipeline components
    joblib.dump(scaler_num, "models/scaler_num.pkl")
    joblib.dump(ohe, "models/ohe.pkl")
    joblib.dump(tfidf, "models/tfidf.pkl")
    joblib.dump(svd, "models/svd.pkl")
    joblib.dump(iso_forest, "models/iso_forest.pkl")
    joblib.dump(scaler_final, "models/scaler_final.pkl")
    print(" - Saved all fitted estimators to 'models/' directory.")
    
    # 9. Plot 2D UMAP Distribution
    print("\n[Step 9] Plotting UMAP 2D feature distribution...")
    plt.figure(figsize=(12, 8))
    # Plot a sample of 50,000 points (instead of 20,000) with optimized size and alpha to prevent overplotting
    plot_sample_size = min(50000, len(train_2d))
    plot_indices = np.random.choice(len(train_2d), plot_sample_size, replace=False)
    
    scatter = plt.scatter(
        train_2d[plot_indices, 0], 
        train_2d[plot_indices, 1], 
        s=3, 
        alpha=0.35, 
        c=df_train_clean.iloc[plot_indices]['salary_min_m_vnd'],
        cmap='viridis',
        edgecolors='none'
    )
    plt.colorbar(scatter, label='Mức lương tối thiểu (Triệu VND/tháng)')
    plt.title('Bản đồ chiếu UMAP 2D - Phân bổ mức lương tuyển dụng', fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Thành phần 1')
    plt.ylabel('UMAP Thành phần 2')
    plt.tight_layout()
    plt.savefig("plots/feature_distribution_2d.png", dpi=150)
    plt.close()
    print(" - Saved visualization plot to 'plots/feature_distribution_2d.png'.")
    
    # Statistics Summary
    print("\n==================================================")
    print("PHASE 2 SUMMARY STATISTICS REPORT")
    print("==================================================")
    print(f"Raw Structured columns count: {len(num_cols) + len(cat_cols)}")
    print(f"Scaled Structured columns count: {struct_train.shape[1]}")
    print(f"SVD Text components: {text_train.shape[1]}")
    print(f"Combined features dimension (before UMAP): {full_train.shape[1]}")
    print(f"Outliers detected & removed: {np.sum(~inliers_mask)} ({np.mean(~inliers_mask)*100:.2f}%)")
    print(f"Train Cluster features shape: {train_20d.shape}")
    print(f"Train Viz features shape: {train_2d.shape}")
    print(f"Test Cluster features shape: {test_20d.shape}")
    print(f"Test Viz features shape: {test_2d.shape}")
    print(f"Has NaNs in Train Cluster: {np.isnan(train_20d).any()}")
    print(f"Has NaNs in Test Cluster: {np.isnan(test_20d).any()}")
    print(f"Total Phase 2 execution time: {time.time() - t_start:.2f} seconds.")
    print("==================================================")

if __name__ == "__main__":
    process_pipeline()
