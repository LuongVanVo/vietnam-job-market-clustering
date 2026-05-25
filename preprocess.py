import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def clean_text(text):
    """
    Cleans raw text:
    - Normalizes HTML tags
    - Removes URLs and emails
    - Removes phone numbers (Vietnamese format)
    - Normalizes whitespace and removes special characters except Vietnamese accents
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase the text
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove Email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove phone numbers (e.g., +84987654321, 0901234567, 090 123 4567)
    text = re.sub(r'(?:\+84|0)(?:\s*\d){9,10}', ' ', text)
    
    # Keep only alphanumeric, whitespace, and Vietnamese accented characters
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    
    # Normalize multiple spaces into single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def parse_salary_string(s):
    """
    Parses a salary string and returns (min_salary, max_salary) in Million VND/month.
    """
    if not isinstance(s, str):
        return np.nan, np.nan
    s = s.lower().strip()
    if s in ['đang cập nhật', 'cạnh tranh', 'thỏa thuận', 'thương lượng', '']:
        return np.nan, np.nan
    
    # Remove thousands separators inside numbers
    s = re.sub(r'(?<=\d)[\.,](?=\d)', '', s)
    
    # Identify currency/unit
    unit = 'vnd'
    if 'usd' in s or '$' in s:
        unit = 'usd'
    elif 'triệu' in s or 'tr/tháng' in s or 'tr' in s:
        unit = 'trieu'
    elif 'nghìn' in s or 'k' in s or 'ngàn' in s:
        if 'nghìn/tuần' in s or 'k/tuần' in s or 'nghìn/ tuần' in s:
            unit = 'nghin_tuan'
        else:
            unit = 'nghin'
            
    # Extract digits
    numbers = re.findall(r'\d+', s)
    if not numbers:
        return np.nan, np.nan
    
    nums = [float(n) for n in numbers]
    
    # Determine min and max
    if len(nums) >= 2:
        val_min, val_max = nums[0], nums[1]
    else:
        val_min, val_max = nums[0], nums[0]
        
    # Scale values to Million VND
    if unit == 'trieu':
        pass
    elif unit == 'usd':
        val_min = val_min * 25000 / 1e6
        val_max = val_max * 25000 / 1e6
    elif unit == 'nghin_tuan':
        val_min = val_min * 1000 * 4.33 / 1e6
        val_max = val_max * 1000 * 4.33 / 1e6
    elif unit == 'nghin':
        val_min = val_min * 1000 / 1e6
        val_max = val_max * 1000 / 1e6
    else:
        if val_min >= 1000:
            val_min = val_min / 1e6
        if val_max >= 1000:
            val_max = val_max / 1e6
            
    # Basic sanity bounds check
    if val_min > 500 or val_max > 500:
        return np.nan, np.nan
        
    return val_min, val_max

def clean_location(loc):
    """
    Cleans raw location string to extract city/province name and group rare ones.
    """
    if not isinstance(loc, str):
        return "Khác"
    loc_lower = loc.lower().strip()
    
    # Common mappings for major cities/regions
    if "hồ chí minh" in loc_lower or "hcm" in loc_lower or "sài gòn" in loc_lower or "sai gon" in loc_lower or "phú nhuận" in loc_lower or "tân bình" in loc_lower or "quận 1" in loc_lower or "quận 3" in loc_lower or "quận 7" in loc_lower or "thủ đức" in loc_lower or "gia định" in loc_lower or "bến nghệ" in loc_lower:
        return "Hồ Chí Minh"
    if "hà nội" in loc_lower or "ha noi" in loc_lower or "cầu giấy" in loc_lower or "hoài đức" in loc_lower or "đống đa" in loc_lower or "ba đình" in loc_lower or "thanh xuân" in loc_lower or "hai bà trưng" in loc_lower or "nam từ liêm" in loc_lower or "bắc từ liêm" in loc_lower or "hoàn kiếm" in loc_lower or "hà đông" in loc_lower:
        return "Hà Nội"
    if "đà nẵng" in loc_lower or "da nang" in loc_lower:
        return "Đà Nẵng"
    if "bình dương" in loc_lower or "binh duong" in loc_lower:
        return "Bình Dương"
    if "đồng nai" in loc_lower or "dong nai" in loc_lower or "biên hòa" in loc_lower:
        return "Đồng Nai"
    if "bắc ninh" in loc_lower or "bac ninh" in loc_lower:
        return "Bắc Ninh"
    if "hải phòng" in loc_lower or "hai phong" in loc_lower:
        return "Hải Phòng"
    if "cần thơ" in loc_lower or "can tho" in loc_lower:
        return "Cần Thơ"
    if "hưng yên" in loc_lower or "hung yen" in loc_lower:
        return "Hưng Yên"
    if "long an" in loc_lower or "tân an" in loc_lower:
        return "Long An"
    if "khánh hòa" in loc_lower or "khanh hoa" in loc_lower or "nha trang" in loc_lower:
        return "Khánh Hòa"
    if "vũng tàu" in loc_lower or "vung tau" in loc_lower or "bà rịa" in loc_lower:
        return "Bà Rịa - Vũng Tàu"
    if "hải dương" in loc_lower or "hai duong" in loc_lower:
        return "Hải Dương"
    if "quảng ninh" in loc_lower or "quang ninh" in loc_lower:
        return "Quảng Ninh"
        
    # Split by comma/pipe to check last part
    parts = re.split(r'[,|\|]', loc)
    if len(parts) > 1:
        for part in reversed(parts):
            p = part.strip()
            p_clean = re.sub(r'^(tỉnh|thành phố|tp\.?|tphc\.?)\s+', '', p, flags=re.IGNORECASE).strip()
            p_clean = re.sub(r'\s+việt nam$', '', p_clean, flags=re.IGNORECASE).strip()
            p_clean = re.sub(r'^việt nam$', '', p_clean, flags=re.IGNORECASE).strip()
            p_lower = p_clean.lower()
            if p_lower in ['hồ chí minh', 'hà nội', 'đà nẵng', 'bình dương', 'đồng nai', 'bắc ninh', 'hải phòng', 'cần thơ', 'hưng yên', 'long an', 'khánh hòa', 'hải dương', 'quảng ninh']:
                continue
            if len(p_clean) > 2 and len(p_clean) < 25 and not any(char.isdigit() for char in p_clean):
                return p_clean.title()
                
    return "Khác"


def parse_experience_string(s):
    """
    Parses experience level string into min and max years.
    """
    if not isinstance(s, str):
        return np.nan, np.nan
    s = s.lower().strip()
    if s in ['không', 'không yêu cầu', 'chưa có kinh nghiệm', '']:
        return 0.0, 0.0
    if 'dưới 1 năm' in s:
        return 0.0, 1.0
    if 'trên' in s:
        nums = re.findall(r'\d+', s)
        if nums:
            val = float(nums[0])
            return val, val + 2.0
        
    numbers = re.findall(r'\d+', s)
    if not numbers:
        return np.nan, np.nan
    
    nums = [float(n) for n in numbers]
    if len(nums) >= 2:
        return nums[0], nums[1]
    else:
        return nums[0], nums[0]

def build_preprocessing_pipeline(sample_size=None):
    print("==================================================")
    print("PHASE 1: DATA PREPROCESSING AND CLEANING STARTING")
    print("==================================================")
    
    # Create data and plots directory if not exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Load dataset from HF
    print("\n[Step 1] Loading dataset 'tinixai/vietnamese-job-descriptions' from HF...")
    ds = load_dataset('tinixai/vietnamese-job-descriptions', split='train')
    
    # Check if sampling is requested
    if sample_size is not None and sample_size < len(ds):
        print(f"Sampling {sample_size} rows for speed and disk optimization.")
        df_raw = pd.DataFrame(ds.select(range(sample_size)))
    else:
        print(f"Using full dataset of {len(ds)} rows.")
        df_raw = pd.DataFrame(ds)
        
    print(f"Dataset shape: {df_raw.shape}")
    
    # 2. Train / Test Split (90/10)
    print("\n[Step 2] Splitting dataset into train (90%) and test (10%)...")
    df_raw_train, df_raw_test = train_test_split(df_raw, test_size=0.1, random_state=42)
    print(f"Raw Train Shape: {df_raw_train.shape} | Raw Test Shape: {df_raw_test.shape}")
    
    # Save raw datasets to data/ folder
    print("Saving raw split files to data/ folder...")
    df_raw_train.to_csv("data/raw_data_train.csv", index=False)
    df_raw_test.to_csv("data/raw_data_test.csv", index=False)
    print("Saved 'data/raw_data_train.csv' and 'data/raw_data_test.csv' successfully.")
    
    # 3. Preprocessing logic
    print("\n[Step 3] Processing train and test sets...")
    
    # Define categorical and text columns
    text_cols = ['job_title', 'job_description', 'benefits', 'requirements']
    cat_cols = ['location', 'job_type', 'job_industry', 'experience_level', 'education_level', 'job_position']
    
    # Compute modes from raw train set to fill missing categories (prevents data leakage)
    cat_modes = {col: df_raw_train[col].mode()[0] if not df_raw_train[col].dropna().empty else "Khác" for col in cat_cols}
    
    # Parse salaries and experience for train raw
    temp_sal_min, temp_sal_max = zip(*df_raw_train['salary'].apply(parse_salary_string))
    temp_exp_min, temp_exp_max = zip(*df_raw_train['experience_level'].apply(parse_experience_string))
    
    # Compute stats for capping & imputation (on train only)
    valid_sal_min = [s for s in temp_sal_min if not np.isnan(s)]
    median_sal_min = np.median(valid_sal_min) if valid_sal_min else 9.0
    median_sal_max = np.median([s for s in temp_sal_max if not np.isnan(s)]) if valid_sal_min else 15.0
    median_exp_min = np.nanmedian(temp_exp_min) if not np.all(np.isnan(temp_exp_min)) else 3.0
    median_exp_max = np.nanmedian(temp_exp_max) if not np.all(np.isnan(temp_exp_max)) else 3.0
    
    # IQR for salary outlier capping
    if valid_sal_min:
        q75, q25 = np.percentile(valid_sal_min, 75), np.percentile(valid_sal_min, 25)
        iqr = q75 - q25
        sal_cap_lower = max(0.0, q25 - 1.5 * iqr)
        sal_cap_upper = q75 + 3.0 * iqr
    else:
        sal_cap_lower, sal_cap_upper = 0.0, 23.0
        
    print(f"Computed Imputation/Capping constants from Train Set:")
    print(f" - Salary Min Median: {median_sal_min:.2f}M | Max Median: {median_sal_max:.2f}M")
    print(f" - Exp Min Median: {median_exp_min:.2f} yrs | Max Median: {median_exp_max:.2f} yrs")
    print(f" - Salary Capping Upper Threshold (IQR 3.0): {sal_cap_upper:.2f}M")
    
    def process_df(df_input, is_train=True):
        df = df_input.copy()
        
        # A. Fill missing strings
        for col in text_cols:
            df[col] = df[col].fillna("")
        for col in cat_cols:
            df[col] = df[col].fillna(cat_modes[col])
            
        # B. Clean texts
        for col in text_cols:
            df[col] = df[col].apply(clean_text)
        
        # Clean location
        df['location'] = df['location'].apply(clean_location)
            
        # C. Concatenate texts
        df['text_combined'] = df['job_title'] + " " + df['requirements'] + " " + df['job_description'] + " " + df['benefits']
        df['text_combined'] = df['text_combined'].str.strip()
        df['word_count'] = df['text_combined'].apply(lambda x: len(x.split()))
        
        # D. Parse salary & experience
        df['salary_min_m_vnd'], df['salary_max_m_vnd'] = zip(*df['salary'].apply(parse_salary_string))
        df['exp_min_years'], df['exp_max_years'] = zip(*df['experience_level'].apply(parse_experience_string))
        
        # E. Capture metrics BEFORE outlier handling
        if is_train:
            nonlocal raw_word_counts, raw_salaries_min
            raw_word_counts = df['word_count'].copy()
            raw_salaries_min = df['salary_min_m_vnd'].copy()
            
        # F. Handle outliers
        before_len = len(df)
        df = df[(df['word_count'] >= 50) & (df['word_count'] <= 5000)].copy()
        after_text_len = len(df)
        
        # Salary outlier: cap values to [sal_cap_lower, sal_cap_upper]
        df['salary_min_m_vnd'] = df['salary_min_m_vnd'].clip(lower=sal_cap_lower, upper=sal_cap_upper)
        df['salary_max_m_vnd'] = df['salary_max_m_vnd'].clip(lower=sal_cap_lower, upper=sal_cap_upper * 1.5)
        
        # G. Impute numerical NaNs with Train Medians
        df['salary_min_m_vnd'] = df['salary_min_m_vnd'].fillna(median_sal_min)
        df['salary_max_m_vnd'] = df['salary_max_m_vnd'].fillna(median_sal_max)
        df['exp_min_years'] = df['exp_min_years'].fillna(median_exp_min)
        df['exp_max_years'] = df['exp_max_years'].fillna(median_exp_max)
        
        print(f"Processed {'Train' if is_train else 'Test'} set: Size before clean: {before_len} | Size after clean: {after_text_len} | Removed: {before_len - after_text_len} text outliers ({((before_len - after_text_len)/before_len)*100:.2f}%)")
        return df

    raw_word_counts = pd.Series()
    raw_salaries_min = pd.Series()
    
    df_clean_train = process_df(df_raw_train, is_train=True)
    df_clean_test = process_df(df_raw_test, is_train=False)
    
    # Save clean datasets to data/ folder
    print("Saving clean split files to data/ folder...")
    df_clean_train.to_csv("data/clean_data_train.csv", index=False)
    df_clean_test.to_csv("data/clean_data_test.csv", index=False)
    print("Saved 'data/clean_data_train.csv' and 'data/clean_data_test.csv' successfully.")
    
    # 4. Generate comparison plots
    print("\n[Step 4] Generating comparison plots...")
    
    # Plot 1: Word Count Comparison
    plt.figure()
    sns.histplot(raw_word_counts, bins=100, color='red', alpha=0.5, label='Raw (Before)', kde=True)
    sns.histplot(df_clean_train['word_count'], bins=100, color='blue', alpha=0.5, label='Clean (After)', kde=True)
    plt.xlim(0, 1000)
    plt.title('Text Word Count Distribution (Before vs After Cleaning)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/word_count_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: Salary Box Plot
    plt.figure(figsize=(10, 6))
    salary_data = pd.DataFrame({
        'Salary': pd.concat([raw_salaries_min.dropna(), df_clean_train['salary_min_m_vnd']]),
        'Group': ['Raw (Before)'] * len(raw_salaries_min.dropna()) + ['Clean (After)'] * len(df_clean_train)
    })
    sns.boxplot(x='Group', y='Salary', data=salary_data, palette='Set2')
    plt.title('Salary Distribution in Million VND (Before vs After Outlier Capping)')
    plt.ylabel('Salary (Million VND / Month)')
    plt.tight_layout()
    plt.savefig('plots/salary_comparison.png', dpi=150)
    plt.close()

    print("Saved comparison plots to 'plots/' directory:")
    print(" - plots/word_count_comparison.png")
    print(" - plots/salary_comparison.png")

    # 5. Outlier and Outline Reports
    print("\n==================================================")
    print("PHASE 1 ANALYSIS: OUTLIER & OUTLINE METRICS REPORT")
    print("==================================================")
    
    print("\n--- WORD COUNT STATS ---")
    print(f"Raw Word Count Range: {raw_word_counts.min()} to {raw_word_counts.max()} words")
    print(f"Clean Word Count Range: {df_clean_train['word_count'].min()} to {df_clean_train['word_count'].max()} words")
    
    print("\n--- SALARY STATS (Million VND / Month) ---")
    print(f"Raw Salary Min parsed range: {raw_salaries_min.min():.2f}M to {raw_salaries_min.max():.2f}M")
    print(f"Clean Salary Min (capped & imputed) range: {df_clean_train['salary_min_m_vnd'].min():.2f}M to {df_clean_train['salary_min_m_vnd'].max():.2f}M")
    
    print("\n--- EXPERIENCE STATS (Years) ---")
    print(f"Clean Exp Min range: {df_clean_train['exp_min_years'].min():.1f} to {df_clean_train['exp_min_years'].max():.1f} years")

    # Shortest and longest text analysis
    print("\n--- TEXT EXTREMES (OUTLIERS DEMONSTRATION) ---")
    
    # Get shortest raw text combined
    raw_texts = df_raw_train['job_title'].fillna("") + " " + df_raw_train['job_description'].fillna("")
    raw_texts_len = raw_texts.str.len()
    
    shortest_raw_idx = raw_texts_len.idxmin()
    longest_raw_idx = raw_texts_len.idxmax()
    
    print("\nShortest Raw Combined Text (Title + Description):")
    print(f"Index: {shortest_raw_idx} | Length: {raw_texts_len.min()} characters")
    print(f"Content: {repr(raw_texts.loc[shortest_raw_idx])[:300]}")
    
    print("\nLongest Raw Combined Text (Title + Description):")
    print(f"Index: {longest_raw_idx} | Length: {raw_texts_len.max()} characters")
    print(f"Content (Truncated to first 400 chars):")
    print(f"{raw_texts.loc[longest_raw_idx][:400]}...")
    
    # Shortest and longest clean text combined
    clean_texts_len = df_clean_train['text_combined'].str.len()
    shortest_clean_idx = clean_texts_len.idxmin()
    longest_clean_idx = clean_texts_len.idxmax()
    
    print("\nShortest Cleaned Combined Text:")
    print(f"Index: {shortest_clean_idx} | Length: {clean_texts_len.min()} characters | Word Count: {df_clean_train.loc[shortest_clean_idx, 'word_count']}")
    print(f"Content: {repr(df_clean_train.loc[shortest_clean_idx, 'text_combined'])[:300]}")
    
    print("\nLongest Cleaned Combined Text:")
    print(f"Index: {longest_clean_idx} | Length: {clean_texts_len.max()} characters | Word Count: {df_clean_train.loc[longest_clean_idx, 'word_count']}")
    print(f"Content (Truncated to first 400 chars):")
    print(f"{df_clean_train.loc[longest_clean_idx, 'text_combined'][:400]}...")
    
    print("\n==================================================")
    print("PHASE 1 COMPLETE: ALL OUTPUTS GENERATED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 1 Preprocessing and Cleaning')
    parser.add_argument('--sample', type=int, default=None, help='Sample size to load (default: full dataset)')
    args = parser.parse_args()
    
    build_preprocessing_pipeline(sample_size=args.sample)
