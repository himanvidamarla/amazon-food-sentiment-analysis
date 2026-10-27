"""
Sentiment Analysis
[On Amazon Fine Food Reviews]
Advanced Text and Social Media Analytics Lab
By Himanvi(23BCB7174), Srinija(23BCB7193), Sheryln(23BCB7296)
"""

from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import nltk
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

DATA_PATH = Path(r"C:\Users\Admin\OneDrive\Desktop\Code\CSE4039\senti-analysis\archive (1)\Reviews.csv")
SAMPLE_SIZE = 2000
RUN_ROBERTA = True
RANDOM_STATE = 42

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}. Update DATA_PATH.")
df = pd.read_csv(DATA_PATH)
if not {"Score", "Text"}.issubset(df.columns):
    raise ValueError("CSV must contain 'Score' and 'Text' columns.")
keep = [c for c in ["Id", "Score", "Text", "Summary", "ProductId", "UserId"] if c in df.columns]
df = df[keep].dropna(subset=["Score", "Text"]).copy()
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df = df[df["Score"].between(1, 5)].copy()
df["Score"] = df["Score"].astype(int)
df["Text"] = df["Text"].astype(str).str.strip()
df = df[df["Text"].ne("")].copy()
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
df = df.reset_index(drop=True)
if "Id" not in df:
    df["Id"] = df.index + 1

print("Rows analysed:", len(df))
print(df.head())

ax = df["Score"].value_counts().sort_index().plot(kind="bar", title="Number of Reviews by Star Rating")
ax.set_xlabel("Star rating")
ax.set_ylabel("Review count")
plt.tight_layout()
plt.show()

df["word_count"] = df["Text"].str.split().str.len()
sns.histplot(data=df, x="word_count", bins=30)
plt.title("Review Length in Words")
plt.tight_layout()
plt.show()

for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "vader_lexicon"]:
    nltk.download(resource, quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
vader_records = []
for row in tqdm(df[["Id", "Text"]].itertuples(index=False), total=len(df), desc="VADER"):
    vader_records.append({"Id": row.Id, **{f"vader_{k}": v for k, v in sia.polarity_scores(row.Text).items()}})
results_df = df.merge(pd.DataFrame(vader_records), on="Id", how="left")

sns.barplot(data=results_df, x="Score", y="vader_compound", errorbar=None)
plt.title("Average VADER Compound Score by Star Rating")
plt.tight_layout()
plt.show()

if RUN_ROBERTA:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def roberta_sentiment(text):
        encoded = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0].detach().cpu().numpy()
        probs = softmax(logits)
        return {"roberta_neg": float(probs[0]), "roberta_neu": float(probs[1]), "roberta_pos": float(probs[2])}

    model_records = []
    for row in tqdm(df[["Id", "Text"]].itertuples(index=False), total=len(df), desc="RoBERTa"):
        try:
            model_records.append({"Id": row.Id, **roberta_sentiment(row.Text)})
        except (RuntimeError, ValueError) as exc:
            print(f"Skipped Id={row.Id}: {exc}")
    results_df = results_df.merge(pd.DataFrame(model_records), on="Id", how="left")

    sns.barplot(data=results_df, x="Score", y="roberta_pos", errorbar=None)
    plt.title("Average RoBERTa Positive Probability by Star Rating")
    plt.tight_layout()
    plt.show()

    print("\\n1-star review with highest RoBERTa positive probability:")
    print(results_df[results_df.Score == 1].sort_values("roberta_pos", ascending=False)[["Score", "Text", "roberta_pos"]].head(1).to_string(index=False))
    print("\\n5-star review with highest RoBERTa negative probability:")
    print(results_df[results_df.Score == 5].sort_values("roberta_neg", ascending=False)[["Score", "Text", "roberta_neg"]].head(1).to_string(index=False))

results_df.to_csv("amazon_food_sentiment_results.csv", index=False)
print("\\nSaved: amazon_food_sentiment_results.csv")

# Compare sentiment scores from VADER and RoBERTa
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(
    data=results_df, x="Score", y="vader_pos",
    errorbar=None, ax=axes[0]
)
axes[0].set_title("VADER Positive Score by Star Rating")
axes[0].set_xlabel("Star Rating")
axes[0].set_ylabel("Average Positive Score")

sns.barplot(
    data=results_df, x="Score", y="roberta_pos",
    errorbar=None, ax=axes[1]
)
axes[1].set_title("RoBERTa Positive Probability by Star Rating")
axes[1].set_xlabel("Star Rating")
axes[1].set_ylabel("Average Positive Probability")

plt.tight_layout()
plt.show()