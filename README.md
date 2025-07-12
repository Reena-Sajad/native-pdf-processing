# Pipeline: Text Preprocessing and Text Classification in a native PDF

This projet builds an end-to-end natural language processing pipeline from **PDF text extraction** to **binary and multi-class text classification** using deep learning‑based features.

## Project Overview

1. **Extract text** from a PDF using PyMuPDF.
2. **Preprocess text**:
   - Clean HTML and punctuation
   - Convert emojis to text
   - Expand contractions
   - Remove stopwords, profanity, and fix spelling
   - Tokenize, stem, and lemmatize
3. **Vectorize** preprocessed text using:
   - Bag‑of‑Words (BoW)
   - TF‑IDF
   - BERT Embeddings
4. **Text Classification**:
   - **Binary Sentiment Classification** using VADER
   - **Stacking Ensemble Model** (Logistic Regression, Naive Bayes, SVM, Random Forest, Decision Tree)
   - **Evaluation Metrics**: Accuracy, Precision, Recall, F1‑Score, ROC‑AUC
   - Visualizations: ROC Curves, Confusion Matrix, Metric Bar Charts

## Highlights

- Uses **BERT embeddings** for deep semantic representation.
- Employs **StackingClassifier** from `sklearn.ensemble` for robust predictive performance.
- Classifies sentences into **positive** or **negative** based on VADER compound score thresholds.
- Supports extension to **multi‑class problems** by modifying the labeling logic.

## Libraries Used

- `PyMuPDF (fitz)`
- `nltk`, `emoji`, `beautifulsoup4`
- `better_profanity`, `pyspellchecker`
- `sklearn`, `matplotlib`, `seaborn`
- `transformers` (BERT tokenizer & model)
- `torch`

## Files

```
SajadHyderReena_Question1.ipynb  # Full notebook pipeline
Question1.pdf                    # Input PDF (expected in Google Drive)
```

## How to Run (Colab Recommended)

1. Mount your Google Drive.
2. Install all required libraries using the `!pip install` commands.
3. Ensure **Question1.pdf** is located correctly.
4. Run all cells in order for the full preprocessing and classification pipeline.

## Outputs

- Preprocessed sentence examples
- BERT vector representations
- Binary sentiment classification results using stacking ensemble
- Evaluation reports:
  - Accuracy, F1, ROC‑AUC
  - Confusion matrix
  - ROC curve per fold

## Example Models Used

- Logistic Regression
- Gaussian Naive Bayes
- Support Vector Machine
- Random Forest
- Decision Tree
- StackingClassifier (meta‑classifier: Logistic Regression)

## Authors

- **Sajad Hyder**
- **Reena**

## Future Enhancements

- Extend to full multi‑class classification
- Add deep learning model (e.g., LSTM or BERT fine‑tuning)
- Incorporate topic modeling or named entity recognition
