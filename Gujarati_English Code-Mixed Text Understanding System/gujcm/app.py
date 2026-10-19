from flask import Flask, request, jsonify, render_template
from nlp_pipeline import analyze
import models

app = Flask(__name__)
models.train()

def full(text):
    c = models.classify(text)
    return {"analysis": analyze(text), **c}

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    t = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not t: return jsonify(error="text is required"), 400
    return jsonify(full(t[:1000]))

@app.route("/api/batch", methods=["POST"])
def api_batch():
    ts = [x.strip() for x in (request.get_json(silent=True) or {}).get("texts", []) if x.strip()][:200]
    out = []
    for t in ts:
        r = full(t); p = r["predictions"]
        out.append({"text": t, "sentiment": p["sentiment"]["label"], "intent": p["intent"]["label"],
                    "topic": p["topic"]["label"], "cmi": r["analysis"]["cmi"]})
    return jsonify(results=out)

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    d = request.get_json(silent=True) or {}
    if d.get("task") not in models.METRICS or not d.get("label") or not d.get("text"):
        return jsonify(error="task, label, text required"), 400
    models.add_feedback(d["task"], d["label"], d["text"][:1000])
    return jsonify(ok=True, metrics=models.METRICS)

@app.route("/api/metrics")
def api_metrics(): return jsonify(models.METRICS)

if __name__ == "__main__":
    app.run(debug=False)
