import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset

data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the useful columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Convert labels: ham -> 0, spam -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# 2. Split Data

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)


# 3. Build Pipeline (TF-IDF + Model)

spam_detector = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
    ('model', MultinomialNB())
])


# 4. Train Model

spam_detector.fit(X_train, y_train)

# 5. Evaluate Model

y_pred = spam_detector.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 6. Test on New Emails

sample_emails = [
    "Claim your free coupon now! Limited offer!",
    "Hey, are we still meeting for lunch tomorrow?"
]

predictions = spam_detector.predict(sample_emails)

for email, label in zip(sample_emails, predictions):
    print(f"Email: {email}\nPrediction: {'Spam' if label==1 else 'Ham'}\n")
