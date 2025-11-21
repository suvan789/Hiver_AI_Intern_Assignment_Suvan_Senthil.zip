import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# --------- Load Models ---------
with open("models/global_model.pkl", "rb") as f:
    global_model = pickle.load(f)

with open("models/tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Allowed tags per customer
customer_allowed_tags = {
    "CUST_A": ["workflow_issue", "notification_bug", "auth_issue"],
    "CUST_B": ["sync_issue", "analytics_issue", "mobile_issue"]
}

# --------- Guardrails ----------
def apply_guardrails(text, predicted):
    text_lower = text.lower()

    if any(k in text_lower for k in ["workflow", "rule", "assign", "automation"]):
        return "workflow_issue"

    if any(k in text_lower for k in ["notification", "alert"]):
        return "notification_bug"

    if any(k in text_lower for k in ["tag", "tagging"]):
        return "tagging_issue"

    return predicted

# --------- Predict Function ---------
def predict_tag(text, customer_id):
    X = vectorizer.transform([text])
    pred = global_model.predict(X)[0]

    # customer tag filtering
    allowed = customer_allowed_tags.get(customer_id, [])
    if pred not in allowed and len(allowed) > 0:
        pred = allowed[0]

    corrected = apply_guardrails(text, pred)

    return {
        "predicted": pred,
        "corrected": corrected,
        "customer_id": customer_id
    }


# --------- Example ----------
if __name__ == "__main__":
    text = "Unable to configure auto assignment rules"
    res = predict_tag(text, "CUST_A")
    print(res)
