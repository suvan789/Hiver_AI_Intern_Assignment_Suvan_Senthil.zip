# Part C – Search & Retrieval (RAG System)

## 1. Overview
This component retrieves the most relevant historical emails using semantic similarity.  
It implements a simplified RAG backend using SentenceTransformers + FAISS.

---

## 2. Approach

### 2.1 Embedding Model
Model Used:  
`all-MiniLM-L6-v2`

Reasons:
- Fast CPU inference
- 384-dimensional embeddings
- Strong semantic performance

### 2.2 Indexing
- Embeddings stored in a FAISS `IndexFlatIP` index
- Inner Product = Cosine similarity (due to normalization)

---

## 3. Retrieval Pipeline

1. Preprocess text (subject + body)
2. Generate embeddings for all emails
3. Store embeddings in FAISS
4. Encode query
5. Retrieve top-K similar emails
6. Return:
   - matched email text  
   - similarity score  
   - inferred tag  
   - reasoning  
   - alternate results

---

## 4. Example Output

```
{
 "query": "unable to assign emails automatically",
 "top_result": {
     "email": "Auto-assign slow Incoming emails remain unassigned...",
     "score": 0.6906,
     "tag": "automation_delay"
 },
 "confidence": 0.846,
 "reasoning": "Top matched email contains similar context...”
}
```

---

## 5. Error Analysis

### Limitations:
- Very short emails → weak embeddings
- No domain-specific fine-tuning
- Some unrelated emails may have similar wording
- No customer-level isolation (can be added in future)

---

## 6. How To Run

### Notebook:
```
partC_notebook.ipynb
```

### Script:
```
python src/partC_rag.py
```

---

## 7. Production Improvements

1. **Fine-tune embeddings on Hiver-specific ticket data**
2. **Hybrid retrieval (keyword + dense + metadata filtering)**
3. **Add LLM summarization for synthesized responses**

---

## 8. Deliverables Included
- `partC_rag_retrieval_system.pdf`
- `partC_notebook.ipynb`
- `src/partC_rag.py`
- This README.md

