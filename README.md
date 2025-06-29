# Multilingual Translator with Semantic Search and BLEU Evaluation

This project is an end-to-end multilingual NLP application that:
- Automatically detects the input language
- Translates text using Meta’s NLLB-200 model
- Retrieves culturally relevant insights using semantic search on a Sanskrit corpus
- Evaluates translation quality using BLEU scoring

The app is built using **Streamlit** and deployed on **Hugging Face Spaces** for public use.

# Features

-  **Language Detection** — XLM-RoBERTa-based classifier
-  **Translation** — NLLB-200 (Distilled, 600M) multilingual model
-  **Semantic Search** — FAISS + SBERT with curated Sanskrit corpus
-  **BLEU Evaluation** — Compare machine output with human reference
-  **Similarity Plot** — Matplotlib-based visualization
-  **Streamlit UI** — Hosted on Hugging Face Spaces

# Tech Stack

- Python 3.x  
- Streamlit  
- HuggingFace Transformers  
- SentenceTransformers  
- FAISS  
- SacreBLEU  
- Matplotlib  

# Installation & Setup

```bash
# Clone the repo
git clone https://github.com/Jeevitha0204/Multilingual-Translator.git
cd Multilingual-Translator

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

Live Demo
 GitHub Repo: https://github.com/Jeevitha0204/Multilingual-Translator

 Live App on Hugging Face Spaces: https://huggingface.co/spaces/jeevitha-app/Multilingual-Translator-App

How It Works
Paste your input text (e.g., in Hindi, French, Tamil, etc.)

Select a target language code (e.g., eng_Latn, fra_Latn)

Optionally provide a human reference translation

View:

 Detected language
 Translated text
 Top 3 semantic matches from the Sanskrit corpus
 BLEU score

Similarity score visualization
