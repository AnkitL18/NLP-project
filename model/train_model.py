import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


print("Loading dataset...")
df = pd.read_csv("dataset/language_detection.csv")

print("Dataset shape      :", df.shape)
print("Columns            :", df.columns.tolist())
print("Languages found    :", df['Language'].nunique())  

# ── Prepare Features & Labels 
X = df['Text']        
y = df['Language']    


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples   :", len(X_train))
print("Test samples       :", len(X_test))

# ── TF-IDF Vectorizer (Exact same config as your Colab) ────────────
print("\n" + "=" * 55)
print("TF-IDF VECTORIZATION")
print("=" * 55)

tfidf_vectorizer = TfidfVectorizer(
    analyzer="char_wb",      
    ngram_range=(2, 4),      
    max_features=50000,      
    sublinear_tf=True,       
    min_df=1,                
    strip_accents=None       
)

# Fit on training data only, transform both sets
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

print("Vocabulary size    :", len(tfidf_vectorizer.vocabulary_))
print("Training matrix    :", X_train_tfidf.shape)
print("Test matrix        :", X_test_tfidf.shape)

# ── Naive Bayes Model (Exact same config as your Colab) ────────────
print("\n" + "=" * 55)
print("TRAINING MULTINOMIAL NAIVE BAYES MODEL")
print("=" * 55)

nb_model = MultinomialNB(alpha=0.1)   # alpha = Laplace smoothing
nb_model.fit(X_train_tfidf, y_train)

y_pred  = nb_model.predict(X_test_tfidf)
y_proba = nb_model.predict_proba(X_test_tfidf)

print("Model trained successfully!")
print("Classes learned    :", list(nb_model.classes_))
print("Number of classes  :", len(nb_model.classes_))

# ── Save Models as .pkl ────────────────────────────────────────────
print("\n" + "=" * 55)
print("SAVING MODELS")
print("=" * 55)

os.makedirs("saved_model", exist_ok=True)

with open("saved_model/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print("✅ tfidf.pkl saved!")

with open("saved_model/nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)
print("✅ nb_model.pkl saved!")

print("\n")