# Gujarati–English Code-Mixed Text Understanding System (Flask)

Run: `pip install -r requirements.txt && python app.py` → open http://127.0.0.1:5000
API: `POST /api/analyze {"text": "..."}`

Pipeline: tokenize → normalize (slang/spelling/elongation) → word-level language ID (GU/EN/HI/OTHER via
script detection + lexicons + suffix heuristics) → Roman→Gujarati transliteration (rule-based + dictionary)
→ classification (sentiment / intent / topic) with char n-gram TF-IDF + Logistic Regression.
Also reports Code-Mixing Index (CMI) and number of language switches.

Upgrade path: replace models.py classifier with `xlm-roberta-base` / `ai4bharat/IndicBERT` fine-tuned via
HuggingFace; use a real dataset (e.g. SAIL-2017, Dravidian-CodeMix-style Gujarati data, or scraped + hand-labelled posts).
