# 📝 Text Summarization Web App

This repository contains a Flask-based web application that provides two types of text summarization:

- **Extractive summarization** using the `bert-extractive-summarizer` (App2)
- **Abstractive summarization** using Hugging Face's `facebook/bart-large-cnn` model (App3)

Both apps include a simple frontend, SQLite history tracking, and customizable summary length.

## 🚀 Features

### ✅ App2 – Extractive Summarizer (`app2.py`)

- Uses the `bert-extractive-summarizer` package
- Summarizes by extracting the most relevant sentences from the input
- Accessible via `/` and `/history`
- HTML template: `templates/index.html`
- CSS: `static/style.css`

### ✅ App3 – Abstractive Summarizer (`app3.py`)

- Uses Hugging Face API to run `facebook/bart-large-cnn`
- Generates new sentences instead of pulling directly from the text
- HTML template: `templates/index1.html`
- CSS: `static/styles.css`

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
