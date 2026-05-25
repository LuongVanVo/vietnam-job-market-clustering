import pandas as pd
import sys

def print_head(filepath="data/clean_data_train.csv", num_rows=10):
    try:
        print(f"Loading first {num_rows} rows from '{filepath}'...")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        df = pd.read_csv(filepath, nrows=num_rows)
        
        # Display selected columns for a clean view
        display_cols = [
            'job_title', 'salary', 'salary_min_m_vnd', 'salary_max_m_vnd', 
            'location', 'job_type', 'experience_level', 'exp_min_years', 
            'education_level', 'word_count'
        ]
        
        # Check if they exist in the dataframe
        cols_to_show = [col for col in display_cols if col in df.columns]
        
        print("\n--- FIRST 10 ROWS (PREVIEW OF STRUCTURED FIELDS) ---")
        print(df[cols_to_show].to_string(index=True))
        
        if 'text_combined' in df.columns:
            print("\n--- FIRST 10 ROWS TEXT COMBINED SNIPPETS ---")
            for i, text in enumerate(df['text_combined']):
                snippet = text[:150] + "..." if len(text) > 150 else text
                print(f"Row {i}: {repr(snippet)}")
                
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please run the preprocessing script first.")
        sys.exit(1)

if __name__ == "__main__":
    filepath = "data/clean_data_train.csv"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    print_head(filepath)
