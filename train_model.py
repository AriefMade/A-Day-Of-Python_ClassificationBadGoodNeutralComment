import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dataset import data



# Custom text preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords (you may want to customize this for Indonesian)
    # stop_words = set(stopwords.words('indonesian'))  # Use if you have Indonesian stopwords
    # tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load dataset
print("Loading dataset...")
# Convert data to DataFrame
df = pd.DataFrame(data, columns=["komentar", "sifat"])

# Check dataset balance
print("\nDistribusi Kelas:")
class_distribution = df['sifat'].value_counts()
print(class_distribution)
print(f"Total data: {len(df)}")

# Apply preprocessing
print("\nMelakukan preprocessing text...")
df['komentar_clean'] = df['komentar'].apply(preprocess_text)

# Split data with stratification
X = df['komentar_clean']
y = df['sifat']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nJumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

# Create pipeline
print("\nMembuat model pipeline...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('clf', MultinomialNB(alpha=1.0))
])

# Train model
print("\nMelatih model...")
model.fit(X_train, y_train)

# Evaluate model
print("\nEvaluasi model:")
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
print("\nCross-validation (5-fold):")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('model/confusion_matrix.png')
print("\nConfusion matrix saved as 'model/confusion_matrix.png'")

# Save the model
print("\nMenyimpan model...")
joblib.dump(model, 'model/model.pkl')

# Save class labels
joblib.dump(model.classes_, 'model/classes.pkl')

print("\nModel berhasil dilatih dan disimpan.")