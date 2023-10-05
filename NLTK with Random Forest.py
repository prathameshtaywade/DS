import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

dataset = pd.read_excel('NLP.xlsx')

dataset.dropna(subset=['description'], inplace=True)
dataset.dropna(subset=['industry'], inplace=True)

X = dataset['description']
y = dataset['industry']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


X = X.apply(preprocess_text)

unique_labels = y.unique()
y = y.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train_vectors, y_train)

# Continue with the evaluation or other tasks

new_description = pd.read_excel('Input.xlsx')
new_description = new_description.dropna()
id = new_description['DocumentId']
segment = new_description['SegmentName']
new_description = new_description['SegmentDescription']
new_description = new_description.apply(preprocess_text)
new_description_vector = vectorizer.transform(new_description)
predicted_industry = model.predict(new_description_vector)
predictions_df = pd.DataFrame({'DocumentId': id, 'SegmentName' : segment, 'Description': new_description, 'Predicted': predicted_industry})
predictions_df['Validation'] = predictions_df['DocumentId'].astype(str) + predictions_df['SegmentName']
predictions_df.to_excel("Prediction.xlsx", index=False)
