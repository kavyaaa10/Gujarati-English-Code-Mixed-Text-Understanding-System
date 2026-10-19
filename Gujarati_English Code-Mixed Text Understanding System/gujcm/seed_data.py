"""TF-IDF (char n-grams, robust to spelling noise) + Logistic Regression classifiers.
Trained on a small demo dataset at startup; replace DATA with a real corpus (see README)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from nlp_pipeline import analyze

DATA = {
"sentiment": {
 "positive": ["aaje to maja aavi gayi bhai superb movie", "tame bahu saras kaam karyu che", "khub saru che yaar loved it",
  "mane aa song bahu gamyu awesome", "food ekdum mast hatu best restaurant", "thank you bhai tame bahu madad kari",
  "kem cho mitro have to maja j maja che", "aa phone nu camera bov saru che paisa vasool", "congratulations bhai bahu khushi thai",
  "ખૂબ સરસ કામ", "mast che yaar full enjoy karyu", "bahu sundar weather che aaje love it"],
 "negative": ["aa movie bakvas hati paisa waste", "service bahu kharab che never again", "mane bahu gusso aave che aa traffic thi",
  "bov bekar experience hato disappointed", "network nathi aavtu so annoying", "delivery late thai ane food thandu hatu",
  "tame kaam j nathi karyu worst", "phone hang thay che bahu irritating", "mane aa bilkul gamyu nahi boring",
  "bhai dukh thayu bahu sad che", "ખરાબ અનુભવ", "refund nathi malyu cheating karyu"],
 "neutral": ["aaje meeting 5 vagye che", "kal office khulli che ke nahi", "mane address moklo please",
  "station ghar thi 5 km dur che", "aaje monday che", "tame kya raho cho", "order number 4521 che",
  "kal weather kevu rehshe", "hu ahmedabad ma rahu chu", "exam time table share karo", "aa shop kyare khule che",
  "bank 10 vagye khule che"]},
"intent": {
 "greeting": ["kem cho bhai", "hello kem chho", "hi mitro su chale che", "good morning tame kem cho", "namaste hello"],
 "question": ["aa shop kyare khule che", "tame kya raho cho", "price ketlu che", "exam kyare che", "shu thayu bhai"],
 "request": ["mane address moklo please", "please refund karo", "plz help karo", "time table share karo", "call karjo mane"],
 "complaint": ["service bahu kharab che", "delivery late thai", "refund nathi malyu", "network nathi aavtu", "phone hang thay che"],
 "appreciation": ["bahu saras kaam karyu", "thank you bhai", "khub saru che love it", "superb food mast", "congratulations bhai"]},
"topic": {
 "sports": ["cricket match ma india jiti gayu", "aaje no match bahu saras hato", "team ma new player aavyo", "score ketlo thayo", "ipl final kyare che"],
 "entertainment": ["aa movie bahu saras che", "navu song hit thai gayu", "film ma hero bahu mast che", "garba song trending che", "web series jovi joie"],
 "food": ["dhokla ane fafda mast che", "restaurant ma food saru hatu", "pizza order karyu delivery late", "chai ane nasto", "thali ma jamvanu bahu saru"],
 "technology": ["phone nu camera saru che", "laptop hang thay che", "app update karo bug che", "wifi network nathi aavtu", "software login problem che"]}}

_models = {}
def train():
    for task, classes in DATA.items():
        X, y = [], []
        for label, texts in classes.items():
            for t in texts:
                X.append(analyze(t)["normalized_text"]); y.append(label)
        m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                          LogisticRegression(max_iter=1000, C=10))
        m.fit(X, y); _models[task] = m

def classify(text):
    norm = analyze(text)["normalized_text"]
    res = {}
    for task, m in _models.items():
        p = m.predict_proba([norm])[0]
        i = p.argmax()
        res[task] = {"label": m.classes_[i], "confidence": round(float(p[i]), 2),
                     "scores": {c: round(float(s), 2) for c, s in zip(m.classes_, p)}}
    return res
