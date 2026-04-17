import joblib

class ArabicTextClassifier:
    def __init__(self):
        self.model = joblib.load("arabic_model.pkl")
        self.vectorizer = joblib.load("cv_ar.pkl")

    def predict(self, text):
        text_vec = self.vectorizer.transform([text])
        return self.model.predict(text_vec)[0]
