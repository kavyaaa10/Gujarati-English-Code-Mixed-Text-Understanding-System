"""Ensemble features (char 2-5 grams + word 1-2 grams) + negation/emoji features + Logistic Regression.
Honest evaluation: train on generated data, test on the hand-written seed set (different distribution).
Final model also learns from data/feedback.csv (user corrections)."""
import csv, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score
from nlp_pipeline import analyze
from seed_data import DATA as SEED
from data_gen import gen

FB = os.path.join(os.path.dirname(__file__), "data", "feedback.csv")
NEGW = {"nathi", "nahi", "not", "no", "never", "na", "નથી", "dont", "cant", "nhi"}
EMO = {"😊": "emopos", "😍": "emopos", "👍": "emopos", "❤": "emopos", "🔥": "emopos", "😂": "emopos",
       "😡": "emoneg", "😢": "emoneg", "👎": "emoneg", "😠": "emoneg", "😞": "emoneg"}
_models, METRICS = {}, {}

def words_of(text):
    return [t["normalized"] for t in analyze(text)["tokens"] if t["lang"] != "OTHER" or t["normalized"] in EMO]

def build(words):
    ws = [EMO.get(w, w) for w in words]
    extra = []
    for i, w in enumerate(ws):
        if w in NEGW:
            extra += ["not_" + x for x in ws[max(0, i - 2):i] + ws[i + 1:i + 3] if x not in NEGW]
    return " ".join(ws + extra)

def _make():
    return make_pipeline(FeatureUnion([
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True))]),
        LogisticRegression(max_iter=3000, C=8))

def _xy(data):
    X, y = [], []
    for lab, texts in data.items():
        for t in texts: X.append(build(words_of(t))); y.append(lab)
    return X, y

def _feedback():
    d = {"sentiment": {}, "intent": {}, "topic": {}}
    if os.path.exists(FB):
        for r in csv.DictReader(open(FB, encoding="utf-8")):
            if r["task"] in d: d[r["task"]].setdefault(r["label"], []).append(r["text"])
    return d

def train():
    G, F = gen(), _feedback()
    for task in SEED:
        Xg, yg = _xy(G[task]); Xs, ys = _xy(SEED[task])
        m = _make().fit(Xg, yg); p = m.predict(Xs)           # honest test: unseen hand-written set
        METRICS[task] = {"accuracy": round(accuracy_score(ys, p), 3), "macro_f1": round(f1_score(ys, p, average="macro"), 3),
                         "train_size": len(Xg), "test_size": len(Xs), "feedback_used": sum(len(v) for v in F[task].values())}
        Xf, yf = _xy(F[task])
        _models[task] = _make().fit(Xg + Xs + Xf, yg + ys + yf)  # final model uses everything
    return METRICS

def _probs(task, words):
    m = _models[task]; return m.classes_, m.predict_proba([build(words)])[0]

def classify(text):
    words = words_of(text); res = {}
    for task in _models:
        cls, p = _probs(task, words); i = p.argmax()
        infl = {}                                              # occlusion: drop in P(label) when word removed
        for j, w in enumerate(words):
            _, q = _probs(task, words[:j] + words[j + 1:])
            infl[j] = round(float(p[i] - q[i]), 3)
        res[task] = {"label": str(cls[i]), "confidence": round(float(p[i]), 3), "uncertain": bool(p[i] < 0.5),
                     "scores": {str(c): round(float(s), 3) for c, s in zip(cls, p)}, "influence": infl}
    return {"words": words, "predictions": res}

def add_feedback(task, label, text):
    os.makedirs(os.path.dirname(FB), exist_ok=True)
    new = not os.path.exists(FB)
    with open(FB, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["task", "label", "text"])
        w.writerow([task, label, text])
    train()
