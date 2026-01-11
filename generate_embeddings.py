import pandas as pd
from sentence_transformers import SentenceTransformer
import sys
import os

INPUT_FILE = 'gemrate_pokemon_catalog_20260108.csv'
OUTPUT_FILE = 'gemrate_embeddings.parquet'
MODEL_NAME = "BAAI/bge-m3"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        sys.exit(1)
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Initial rows: {len(df)}")

    mask_univ = df['UNIVERSAL_GEMRATE_ID'].notna()
    df.loc[mask_univ, "GEMRATE_ID"] = df.loc[mask_univ, "UNIVERSAL_GEMRATE_ID"]
    df = df.drop_duplicates(subset=['GEMRATE_ID'])
    df['YEAR'] = df['YEAR'].astype(str).str.split('-').str[0].str.strip()
    df = df[["GEMRATE_ID", "YEAR", "SET_NAME", "NAME", "CARD_NUMBER", "PARALLEL"]]
    df = df.fillna("")
    df = df.astype(str)
    df = df.reset_index(drop=True)

    print(f" Done. Rows after cleaning: {len(df)}")

    print("Creating text for embedding...")

    df['embedding_text'] = (
        df['YEAR'] + " " + 
        df['SET_NAME'] + " " + 
        df['NAME'] + " " + 
        df['CARD_NUMBER'] + " " + 
        df['PARALLEL']
    )
    df['embedding_text'] = df['embedding_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device='cuda')
    
    print("Generating embeddings (this may take a while)...")

    embeddings = model.encode(
        df['embedding_text'].tolist(), 
        batch_size=256, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    df['embedding_vector'] = list(embeddings)
    df['gemrate_id'] = df["GEMRATE_ID"]
    output_df = df[['gemrate_id', 'embedding_vector']]
    
    print(f"Saving {len(output_df)} rows to {OUTPUT_FILE}...")

    output_df.to_parquet(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
