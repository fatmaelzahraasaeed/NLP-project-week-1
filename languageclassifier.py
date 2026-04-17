import joblib

class LangClassifier:
    def __init__(self):
        self.model = joblib.load("lang_model.pkl")
        self.vectorizer = joblib.load("cv_lang.pkl")

    def predict(self, text):
        text_vec = self.vectorizer.transform([text])
        return self.model.predict(text_vec)[0]
    