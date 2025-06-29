# Import Libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sacrebleu import corpus_bleu
import os

# Load Models
lang_detect_model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
lang_detect_tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Language Code Mappings
id2lang = lang_detect_model.config.id2label

nllb_langs = {
    "eng_Latn": "English", "fra_Latn": "French", "hin_Deva": "Hindi",
    "spa_Latn": "Spanish", "deu_Latn": "German", "tam_Taml": "Tamil",
    "tel_Telu": "Telugu", "jpn_Jpan": "Japanese", "zho_Hans": "Chinese",
    "arb_Arab": "Arabic", "san_Deva": "Sanskrit"
}

xlm_to_nllb = {
    "en": "eng_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "es": "spa_Latn", "de": "deu_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "ja": "jpn_Jpan", "zh": "zho_Hans", "ar": "arb_Arab",
    "sa": "san_Deva"
}

# Static Corpus
corpus = [
    "धर्म एव हतो हन्ति धर्मो रक्षति रक्षितः",
    "Dharma when destroyed, destroys; when protected, protects.",
    "The moon affects tides and mood, according to Jyotisha",
    "One should eat according to the season – Rituacharya",
    "Balance of Tridosha is health – Ayurveda principle",
    "Ethics in Mahabharata reflect situational dharma",
    "Meditation improves memory and mental clarity",
    "Jyotisha links planetary motion with life patterns"
]
corpus_embeddings = embed_model.encode(corpus, convert_to_numpy=True)
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# Detect Language
def detect_language(text):
    inputs = lang_detect_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = lang_detect_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return id2lang[pred]

# Translate
def translate(text, src_code, tgt_code):
    trans_tokenizer.src_lang = src_code
    encoded = trans_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    try:
        target_lang_id = trans_tokenizer.convert_tokens_to_ids([tgt_code])[0]
        generated = trans_model.generate(**encoded, forced_bos_token_id=target_lang_id)
        return trans_tokenizer.decode(generated[0], skip_special_tokens=True)
    except:
        return ""

# Semantic Search
def search_semantic(query, top_k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [(corpus[i], float(distances[0][idx])) for idx, i in enumerate(indices[0])]

# Full pipeline for Gradio
def full_pipeline(user_input_text, target_lang_code, human_ref=""):
    if not user_input_text.strip():
        return "⚠️ Empty input", "", [], "", ""

    detected_lang = detect_language(user_input_text)
    src_nllb = xlm_to_nllb.get(detected_lang, "eng_Latn")

    translated = translate(user_input_text, src_nllb, target_lang_code)
    if not translated:
        return detected_lang, "❌ Translation failed", [], "", ""

    sem_results = search_semantic(translated)
    result_list = [f"{i+1}. {txt} (Score: {score:.2f})" for i, (txt, score) in enumerate(sem_results)]

    # Plot
    labels = [f"{i+1}" for i in range(len(sem_results))]
    scores = [score for _, score in sem_results]
    plt.figure(figsize=(6, 4))
    bars = plt.barh(labels, scores, color="lightgreen")
    plt.xlabel("Similarity Score")
    plt.title("Top Semantic Matches")
    plt.gca().invert_yaxis()
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + 0.1, f"{bar.get_width():.2f}", fontsize=8)
    plt.tight_layout()
    plot_path = "/tmp/sem_plot.png"
    plt.savefig(plot_path)
    plt.close()

    bleu_score = ""
    if human_ref.strip():
        bleu = corpus_bleu([translated], [[human_ref]])
        bleu_score = f"{bleu.score:.2f}"

    return detected_lang, translated, result_list, plot_path, bleu_score

# Gradio App
gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Textbox(label="Input Text", lines=4, placeholder="Enter text to translate..."),
        gr.Dropdown(label="Target Language", choices=list(nllb_langs.keys()), value="eng_Latn"),
        gr.Textbox(label="(Optional) Human Reference Translation", lines=2, placeholder="Paste human translation here (for BLEU)...")
    ],
    outputs=[
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="Translated Text"),
        gr.Textbox(label="Top Semantic Matches"),
        gr.Image(label="Semantic Similarity Plot"),
        gr.Textbox(label="BLEU Score")
    ],
    title="🌍 Multilingual Translator + Semantic Search",
    description="Detects language → Translates → Finds related Sanskrit concepts → BLEU optional."
).launch()
