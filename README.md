# Hiver — AI Intern Assignment (Suvan Senthil)

**Repository:** Hiver AI Intern Assignment  
**Candidate:** Suvan Senthil  
**Position:** AI Intern (6 Months)

---

## Overview

This repository contains my solution to the Hiver AI intern assignment. The project is split into three parts, each implemented locally with notebooks, runnable scripts, and documentation:

- **Part A – Email Tagging**  
  TF-IDF + Logistic Regression baseline, customer-specific models, customer isolation, and rule-based guardrails.

- **Part B – Sentiment Analysis**  
  Local transformer-based sentiment classification (CardiffNLP RoBERTa). Two prompt simulations: V1 (baseline) and V2 (improved with evidence and confidence calibration).

- **Part C – Search & Retrieval (RAG)**  
  SentenceTransformer embeddings + FAISS index for semantic retrieval of past emails. RAG-style output with top-k matches, confidence, and reasoning.

---

## Assignment (Original)

Original assignment uploaded:  
`/mnt/data/AI_Intern_Assignment_1763630504.pdf`

> *(This path points to the uploaded assignment PDF — it will be transformed to a URL by the submission system.)*

---

## Repository Structure

├── deliverables/
│ ├── PartA/
│ │ ├── partA_email_tagging.pdf
│ │ ├── partA_notebook.ipynb
│ │ ├── README.md
│ │ └── src/
│ │ └── partA_predict.py
│ ├── PartB/
│ │ ├── partB_sentiment_analysis.pdf
│ │ ├── partB_notebook.ipynb
│ │ ├── README.md
│ │ └── src/
│ │ └── partB_sentiment.py
│ └── PartC/
│ ├── partC_rag_retrieval_system.pdf
│ ├── partC_notebook.ipynb
│ ├── README.md
│ └── src/
│ └── partC_rag.py
├── notebooks/ # working notebooks (same as deliverables/notebooks)
├── src/ # optional top-level scripts
├── data/ # (not included in ZIP) pointers to datasets
├── requirements.txt
└── README.md # <-- you are reading this
Install dependencies

pip install --upgrade pip
pip install -r requirements.txt


Part A — Email Tagging

Notebook: deliverables/PartA/partA_notebook.ipynb

Script (inference example):

python deliverables/PartA/src/partA_predict.py


This script demonstrates a sample prediction and guardrail correction. See the Part A PDF and README for details on training and reproducing experiments.

Part B — Sentiment

Notebook: deliverables/PartB/partB_notebook.ipynb

Script (local inference):

python deliverables/PartB/src/partB_sentiment.py


This will run the local RoBERTa-based sentiment example on sample inputs. The notebook runs the 10 required dataset rows and compares Prompt V1 vs V2.

Part C — RAG / Retrieval

Notebook: deliverables/PartC/partC_notebook.ipynb

Script (build index + search):

python deliverables/PartC/src/partC_rag.py


This builds a FAISS index from emails and runs a sample query. See the Part C README for configuration.

Requirements

Main dependencies are listed in requirements.txt. Example content:

pandas
scikit-learn
scipy
transformers
torch
sentence-transformers
faiss-cpu
joblib


(Install via pip install -r requirements.txt)

If running on CPU-only machines, installing the CPU wheel of torch may be necessary:

pip install torch --index-url https://download.pytorch.org/whl/cpu

Reproducibility & Notes

The notebooks are designed to run end-to-end when the dataset CSVs are placed in data/:

data/small_dataset.csv

data/large_dataset.csv

For Part A we included both a notebook and a minimal inference script (partA_predict.py) to satisfy the “notebook or script runnable end-to-end” requirement.

For Part B and Part C, notebooks contain full explanatory cells; the src/ scripts provide runnable examples for quick evaluation.

Screenshots, confusion matrices, and sample outputs are included in the deliverables/Part*/ folders for easy review.
What’s in the PDFs

Each part PDF contains:

Problem statement and approach

Implementation details

Key code snippets and architecture

Error analysis

Screenshots (confusion matrices / sample outputs)

Production improvements and next steps
