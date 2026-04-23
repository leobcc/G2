import pandas as pd
import numpy as np
import os

def run_eda(filepath="data/deals.csv"):
    df = pd.read_csv(filepath)
    df['title_length'] = df['title'].fillna('').apply(len)
    df['desc_word_count'] = df['description'].fillna('').apply(lambda x: len(str(x).split()))
    df['fine_print_length'] = df['fine_print'].fillna('').apply(len)
    
    features = ['title_length', 'desc_word_count', 'fine_print_length', 'image_quality_score', 'price', 'discount_pct']
    correlations = df[features + ['cvr']].corr()['cvr'].sort_values(ascending=False)
    
    print("Correlation with CVR:")
    print(correlations)
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/analysis_report.txt", "w") as f:
        f.write("Correlations with CVR:\n")
        f.write(correlations.to_string())

if __name__ == "__main__":
    run_eda()