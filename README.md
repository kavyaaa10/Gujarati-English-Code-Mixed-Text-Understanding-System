# Gujarati–English Code-Mixed Text Understanding System

A Flask-based NLP prototype for analyzing Gujarati–English code-mixed text written in Roman Gujarati, English, and Gujarati script. It combines rule-based text processing with machine-learning classifiers to produce language and text-analysis features.

## Features

- Text normalization for common spelling variations and informal language.
- Word-level language identification (GU, EN, HI, OTHER) using script detection, lexicons, and heuristics.
- Rule-based Roman-to-Gujarati transliteration where supported.
- Sentiment classification: positive, negative, neutral.
- Intent classification: greeting, question, request, complaint, appreciation.
- Topic classification: sports, entertainment, food, technology, and politics (the topic model's labels are determined by the training data).
- Code-Mixing Index (CMI), language distribution, and language-switch count.
- JSON endpoints for single-text analysis, batch analysis, feedback, and metrics.

## Technology Stack

- Python
- Flask
- scikit-learn
- NumPy
- TF-IDF features and Logistic Regression

## Project Structure

```text
.
├── app.py              # Flask application and API routes
├── models.py           # Model training, prediction, metrics, and feedback
├── nlp_pipeline.py     # Text normalization, language identification, transliteration, and analysis
├── seed_data.py        # Small hand-written demonstration examples
├── data_gen.py         # Template-based synthetic data generator
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web interface
└── data/               # Runtime feedback data may be written here
```

## Requirements

- Python 3.10 or later is recommended.
- pip

## Installation and Run

1. Download or clone this repository and open a terminal in the project directory.
2. (Recommended) Create and activate a virtual environment.

   **Windows:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   python app.py
   ```
5. Open http://127.0.0.1:5000 in your browser.

The models are trained when the application starts, so the first startup may take a little longer.

## API Endpoints

### Analyze one text

`POST /api/analyze`

JSON request body:
```json
{
  "text": "aa movie bahu saras che"
}
```

### Analyze a batch of texts

`POST /api/batch`

JSON request body:
```json
{
  "texts": ["kem cho bhai", "aa phone bahu kharab che"]
}
```

### View model metrics

`GET /api/metrics`

### Submit feedback

`POST /api/feedback`

JSON request body:
```json
{
  "task": "sentiment",
  "label": "positive",
  "text": "aa movie bahu saras che"
}
```

Use task names and labels that are supported by the application. Feedback is written to `data/feedback.csv` and the models are retrained. Do not submit private or sensitive text.

## Dataset Information

This repository currently uses two kinds of **demonstration data**:

1. **Hand-written seed examples (`seed_data.py`):** A small, manually curated set of example sentences with labels for sentiment, intent, and topic classification.
2. **Synthetic examples (`data_gen.py`):** A template-based generator that creates additional Gujarati–English code-mixed sentences. These examples are generated programmatically and are not collected from real users.

At startup, the application trains models using the generated examples and evaluates them against the hand-written seed examples, then trains the final models using the generated examples, seed examples, and any saved feedback examples. The metrics shown by the application are therefore useful for prototype testing, but they should **not** be treated as evidence of real-world performance on an independent, representative dataset.

No external research dataset is bundled with this repository at present. If a third-party dataset is added, document its name, source URL, licence/usage terms, label definitions, and preprocessing steps here. For academic evaluation, consider collecting or obtaining an appropriately licensed, representative dataset and evaluating on a separate held-out test set.

## Limitations

- The seed examples are small and may not represent the diversity of Gujarati–English code-mixed text.
- Synthetic template-generated sentences can be repetitive and may not reflect natural language use.
- Language identification and transliteration use rules and heuristics, so results may be imperfect.
- Prediction confidence is a model score, not a guarantee that a prediction is correct.
- Reported metrics depend on the included seed examples and evaluation setup.

## Future Improvements

- Add a larger, documented, appropriately licensed Gujarati–English code-mixed dataset.
- Evaluate with a separate human-labelled test set and report precision, recall, macro-F1, and confusion matrices.
- Improve transliteration and language identification with broader vocabulary and validated methods.
- Explore multilingual transformer models, such as IndicBERT or XLM-R, if suitable training data and computing resources are available.

## License

No license has been specified yet. Add a license file if you intend to define how others may reuse or distribute this project.
