import joblib

class EnglishTextClassifier:
    def __init__(self):
        self.model = joblib.load("model.pkl")
        self.vectorizer = joblib.load("vectorizer.pkl")

    def predict(self, text):
        text_vec = self.vectorizer.transform([text])
        return self.model.predict(text_vec)[0]