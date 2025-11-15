"""
PROJECT  — Text Sentiment Analysis (NLP Project)
--------------------------------------------------
Goal:
Classify text reviews into POSITIVE & NEGATIVE sentiment.

What Concepts We Will Cover:
- Pandas Data Loading
- EDA (head, info, describe, null-checks)
- Cleaning Missing Values
- Text Preprocessing (Lowercasing, Removing Punctuations, Stopwords, Lemmatization)
- Feature Engineering: TF-IDF Vectorizer
- Train-Test Split
- Model Training (Logistic Regression, Naive Bayes, SVM)
- Evaluation Metrics (Accuracy, Precision, Recall, F1)
- Confusion Matrix
- Visualization

Dataset: twitter_sentiment_raw.csv
"""

# -----------------------------------------------
# Step 1: Import Libraries
# -----------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as wr
wr.filterwarnings('ignore')

# NLP Libraries
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------------------------
# Step 2: Load Dataset
# -----------------------------------------------
df = pd.read_csv("dataset/nlp/twitter_sentiment_raw.csv")
print(df.head())
print(df.info())
print(df.describe(include='all'))
print(df.isnull().sum())
print(df.columns)

# Some datasets include unnecessary index-like columns
if "id" in df.columns:
    df = df.drop("id", axis=1)

print(df.shape)
print(df['sentiment'].value_counts())  # our target column

# -----------------------------------------------
# Step 3: Handle Missing Values
# -----------------------------------------------
# drop rows where review is missing
df['text'] = df['text'].fillna("")  # or drop — but we fill for EDA completeness
df['sentiment'] = df['sentiment'].fillna(df['sentiment'].mode()[0])

# -----------------------------------------------
# Step 4: Basic EDA
# -----------------------------------------------
show_graphs = True
if show_graphs:
    sns.countplot(x=df['sentiment'])
    plt.title("Sentiment Distribution")
    plt.show()

# -----------------------------------------------
# Step 5: Text Cleaning Function
# -----------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"@\w+|#", "", text)  # remove @mentions and #hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuations
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.strip()  # remove side spaces

    # remove stopwords & lemmatize
    tokens = []
    for word in text.split():
        if word not in stop_words:
            tokens.append(lemmatizer.lemmatize(word))

    return " ".join(tokens)

# Apply text cleaning
df['clean_text'] = df['text'].apply(clean_text)
print("\nCleaned Text Sample:\n", df[['text', 'clean_text']].head())

# -----------------------------------------------
# Step 6: Encode Target Column
# -----------------------------------------------
# positive = 1 , negative = 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(df['sentiment'].value_counts())

# -----------------------------------------------
# Step 7: Train-Test Split
# -----------------------------------------------
X = df['clean_text']
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", len(x_train))
print("Test size:", len(x_test))

# -----------------------------------------------
# Step 8: TF-IDF Vectorization (Feature Engineering)
# -----------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# -----------------------------------------------
# Step 9: Train ML Models
# -----------------------------------------------
lr_clf = LogisticRegression()
nb_clf = MultinomialNB()
svm_clf = SVC(kernel='linear')

lr_clf.fit(x_train_vec, y_train)
nb_clf.fit(x_train_vec, y_train)
svm_clf.fit(x_train_vec, y_train)

# -----------------------------------------------
# Step 10: Predictions
# -----------------------------------------------
lr_pred = lr_clf.predict(x_test_vec)
nb_pred = nb_clf.predict(x_test_vec)
svm_pred = svm_clf.predict(x_test_vec)

# -----------------------------------------------
# Step 11: Evaluation Function
# -----------------------------------------------
def eval_model(name, y_test, pred):
    print(f"\n------- {name} -------")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1-Score:", f1_score(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Evaluate each model
eval_model("Logistic Regression", y_test, lr_pred)
eval_model("Naive Bayes", y_test, nb_pred)
eval_model("SVM Classifier", y_test, svm_pred)

# -----------------------------------------------
# Step 12: Compare All Model Results
# -----------------------------------------------
models = {
    "Logistic Regression": lr_pred,
    "Naive Bayes": nb_pred,
    "SVM Classifier": svm_pred
}

results = []
for name, pred in models.items():
    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred)
    ])

results_df = pd.DataFrame(
    results,
    columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1']
)

print("\nOverall Comparison:\n")
print(results_df)

# Optional Graph
if show_graphs:
    results_df.plot(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1'], kind='bar')
    plt.title("NLP Model Comparison")
    plt.show()
