# Part B – Sentiment Analysis

## 1. Overview
This module classifies the sentiment (positive, negative, neutral) of 10 sample emails using a fully local transformer model.  
Two prompt versions were implemented:
- Prompt V1: Weak baseline
- Prompt V2: Improved prompt with reasoning and evidence

---

## 2. Approach

### 2.1 Dataset Preparation
- Loaded first 10 rows from dataset
- Combined subject + body into a single text field

### 2.2 Local Model
Model Used:  
`cardiffnlp/twitter-roberta-base-sentiment-latest`

Reasons:
- Lightweight
- Runs offline
- Pretrained for sentiment classification

---

## 3. Prompting Strategy

### 3.1 Prompt V1 (Weak Prompt)
- Direct model output
- No evidence
- No reasoning
- Represents a naive LLM prompt

### 3.2 Prompt V2 (Improved Prompt)
- Extracts evidence keywords (e.g., “unable”, “error”, “thanks”)
- Calibrates confidence
- Generates natural language reasoning
- Produces more stable and explainable results

---

## 4. Error Analysis

### Observations:
- Short emails → model leans towards neutral
- Domain mismatch (Twitter → support emails)
- Keyword bias (presence of “not” overly affects predictions)
- Some negative emotions not captured by simple keywords

---

## 5. How To Run

### Notebook:
```
partB_notebook.ipynb
```

### Script:
```
python src/partB_sentiment.py
```

---

## 6. Production Improvements

1. **Fine-tune sentiment model on Hiver data**
2. **Use domain-specific lexicons for support-related sentiment**
3. **Add conversational context (previous messages)**

---

## 7. Deliverables Included
- `partB_sentiment_analysis.pdf`
- `partB_notebook.ipynb`
- `src/partB_sentiment.py`
- This README.md

