from languageclassifier import LangClassifier
from arabicclassifier import ArabicTextClassifier
from englishclassifier import EnglishTextClassifier

lang_model = LangClassifier()
ar_model = ArabicTextClassifier()
en_model = EnglishTextClassifier()

arabic_labels = {
    1: 'negative',
    2: 'positive'
}

english_labels = {
    0 : 'positive',
    1 : 'Neutral',
    2 : 'Negative'
}

text = input("Enter text: ")

lang = lang_model.predict(text)

print("\nDetected Language:", lang)

if lang == "arabic":
    result = ar_model.predict(text)
    print("Arabic Class:", arabic_labels[result])

else:
    result = en_model.predict(text)
    print("English Class:", english_labels[result])