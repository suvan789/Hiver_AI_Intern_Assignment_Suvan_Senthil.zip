import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
labels = ["negative", "neutral", "positive"]


def sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = softmax(outputs.logits[0].numpy())
    return {
        "sentiment": labels[scores.argmax()],
        "confidence": float(scores.max())
    }


if __name__ == "__main__":
    print(sentiment("The update broke my workflow"))
