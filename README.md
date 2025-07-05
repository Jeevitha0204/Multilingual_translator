# Multilingual Translator + Semantic Search (Enhanced)

This project is a smart multilingual translator web app that offers:
 Automatic **language detection**
 High-quality **translation** between multiple Indian and foreign languages
 **Semantic search** to find similar Sanskrit-based concepts
 Optional **BLEU score** evaluation for comparing translations
 A downloadable `.txt` **report** summarizing the output
 Input length handling to avoid translation errors

Developed using **Hugging Face Transformers**, **Sentence Transformers**, **FAISS**, and **Gradio** — and deployable to **Hugging Face Spaces**.
## Input Limit Notice
  **Please enter only up to 3 lines of text or 2000 characters maximum.**  
> If the input is too long, the app will show an error and skip translation.

##  Live Demo

[Click here to view the live app on Hugging Face Spaces](https://huggingface.co/spaces/jeevitha-app/Multilingual-translator)  
 [GitHub Repository](https://github.com/Jeevitha0204/Multilingual_translator)


## Features

Feature with Description 

Language Detection - Auto-identifies input language using `xlm-roberta-base-language-detection` 
Translation - Uses Facebook's `NLLB-200` model for multilingual translation 
Semantic Search - Finds similar Sanskrit concepts using Sentence Transformers + FAISS 
BLEU Score - Optional metric comparing translation vs. human reference 
Semantic Plot - Horizontal bar chart for top 3 similarity scores 
Download Report - Creates a `.txt` summary file (language, translation, scores) 
Error Handling - Empty input / long input shows friendly messages 


## Supported Languages

| Code      | Language   |
|-----------|------------|
| eng_Latn  | English    |
| hin_Deva  | Hindi      |
| tam_Taml  | Tamil      |
| tel_Telu  | Telugu     |
| san_Deva  | Sanskrit   |
| fra_Latn  | French     |
| spa_Latn  | Spanish    |
| deu_Latn  | German     |
| jpn_Jpan  | Japanese   |
| zho_Hans  | Chinese    |
| arb_Arab  | Arabic     |


# 📄 Downloadable Report

The app generates a downloadable `.txt` file with:

- Detected source language
- Translated output
- Semantic matches (with similarity scores)
- BLEU score if human reference is given

 Future Enhancements
 
*Speech-to-text input support
*Text-to-speech audio output
* OCR: Translate text from images
* Add more Indian languages and support for transliteration

Author
Jeevitha Meenakshisundaram
MSc Data Science | Sastra University


License
This project is licensed under the MIT License.





