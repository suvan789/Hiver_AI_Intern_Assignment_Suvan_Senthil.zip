# Part A – Email Tagging System

## 1. Overview
This component predicts the correct issue tag for incoming customer emails.  
The solution combines **ML**, **customer-specific isolation**, and **rule-based guardrails** to handle extremely sparse dataset conditions.

---

## 2. Approach

### 2.1 Dataset Issues
- ~60 unique tags
- Most tags appear only 1–2 times (extremely sparse)
- Short, noisy text
- Uneven tag distribution per customer

Because of this, pure ML alone cannot perform well.

### 2.2 Pipeline Summary
1. Preprocess text (subject + body)
2. Global TF-IDF + Logistic Regression model
3. Per-customer model training (if data ≥ 5 samples)
4. Customer-isolated inference (allowed tag filtering)
5. Guardrails override ML predictions when needed
6. Final structured JSON output

---

## 3. Models Used

### 3.1 Global Model
- TF-IDF (max_features=5000)
- Logistic Regression (multi-class, balanced class weights)

### 3.2 Per-Customer Models
- Trained only when customer has enough samples
- Keeps tags isolated per customer

### 3.3 Guardrails / Rule-Based Layer
Example rules:
- Workflow → `workflow_issue`
- Tagging words → `tagging_issue`
- Notification → `notification_bug`

These rules correct many mistakes caused by sparse data.

---

## 4. Customer Isolation

### 4.1 Tag Whitelist per Customer
Extracted from historical dataset:
```
CUST_A → {workflow_issue, notification_bug, auth_issue}
CUST_B → {sync_issue, analytics_issue, mobile_issue}
```

### 4.2 Model Isolation
If customer has enough data, a private model is used.

### 4.3 Inference Isolation
Even global model outputs are filtered to allowed tags only.

This completely prevents **cross-customer tag leakage**.

---

## 5. Error Analysis

### Key Issues Identified:
- Heavy class imbalance
- Model collapses to frequent tags
- Short emails → limited semantic information
- Overlapping vocabulary across multiple tags
- Confusion matrix shows poor separability

### Guardrail Fixes:
- Semantic keyword rules fix workflow, tagging, notification issues
- Helps reliability when ML confidence is low

---

## 6. How To Run

### Notebook:
```
partA_notebook.ipynb
```

### Script:
```
python src/partA_predict.py
```

---

## 7. Production Improvements

1. **Replace TF-IDF with embeddings (MiniLM/BERT)**
2. **Human-in-the-loop feedback loop**
3. **Hybrid ML + Keyword Rules + Retrieval (Part C)**

---

## 8. Deliverables Included
- `partA_email_tagging.pdf`
- `partA_notebook.ipynb`
- `src/partA_predict.py`
- `screenshots/`
- This README.md

