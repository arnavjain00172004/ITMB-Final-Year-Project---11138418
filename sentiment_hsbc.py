import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "HSBC_2021_stock_headlines_sentiment.csv"
OUTPUT_CSV = "HSBC_2021_Sentiment.csv"

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print(f"[INFO] Reading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
print(f"[INFO] Loaded {len(df)} rows.\n")
print(df[["date", "headline"]].head(3).to_string(index=False))
print()

# Score on summary text — richer context than headline alone.
# Falls back to headline if summary is missing.
texts = df["summary"].fillna(df["headline"]).tolist()

# ── 2. FinBERT ────────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
device_label = "GPU" if device == 0 else "CPU"
print(f"[INFO] Loading FinBERT on {device_label} (first run downloads ~400 MB) ...")

finbert_pipe = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=device,
    top_k=1,
)
print("[INFO] FinBERT ready.\n")

fb_labels, fb_scores = [], []
for text in tqdm(texts, desc="FinBERT"):
    result = finbert_pipe(text[:500], truncation=True)[0][0]
    fb_labels.append(result["label"])
    fb_scores.append(round(result["score"], 6))

df["finbert_label"] = fb_labels
df["finbert_score"] = fb_scores

label_to_int = {"positive": 1, "neutral": 0, "negative": -1}
df["finbert_numeric"] = df["finbert_label"].map(label_to_int)

# ── 3. VADER ──────────────────────────────────────────────────────────────────
print("\n[INFO] Running VADER ...")
analyzer = SentimentIntensityAnalyzer()

vader_rows = []
for text in tqdm(texts, desc="VADER   "):
    s = analyzer.polarity_scores(text)
    vader_rows.append({
        "vader_compound": round(s["compound"], 6),
        "vader_pos":      round(s["pos"],      6),
        "vader_neg":      round(s["neg"],      6),
        "vader_neu":      round(s["neu"],      6),
    })

vader_df = pd.DataFrame(vader_rows)
df = pd.concat([df.reset_index(drop=True), vader_df], axis=1)

def vader_label(c):
    if c >= 0.05:  return "positive"
    if c <= -0.05: return "negative"
    return "neutral"

df["vader_label"]  = df["vader_compound"].apply(vader_label)
df["models_agree"] = df["finbert_label"] == df["vader_label"]

# ── 4. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[INFO] Saved scored dataset → {OUTPUT_CSV}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n── FinBERT distribution ──────────────────")
print(df["finbert_label"].value_counts().to_string())
print("\n── VADER distribution ────────────────────")
print(df["vader_label"].value_counts().to_string())
print(f"\n── Model agreement rate : {df['models_agree'].mean():.1%}")
print(f"── Avg VADER compound   : {df['vader_compound'].mean():.4f}")

print("\n── Sample output (first 5 rows) ──────────────────────────────────────")
cols = ["date", "headline", "finbert_label", "finbert_score", "vader_compound", "models_agree"]
print(df[cols].head(5).to_string(index=False))