import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import json
import torch.nn as nn

save_path = "/root/ClarityG/TTP/result/text_roberta_0.5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobertaClassifier(nn.Module):
    def __init__(self, roberta_model, hidden_size=768, num_labels=10):
        super().__init__()
        self.roberta = roberta_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(pooled)
        return logits
def init_text_model(save_path, model_path=None):

    with open(os.path.join(save_path, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = RobertaTokenizer.from_pretrained(f"{save_path}/tokenizer")

    roberta_encoder = RobertaModel.from_pretrained("/root/ClarityG/TTP/models/roberta-base")
    model = RobertaClassifier(roberta_encoder, hidden_size=768, num_labels=len(label2id))
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, tokenizer, id2label


def predict_texts(texts, model, tokenizer, id2label, threshold=0.4):

    results = []
    with torch.no_grad():
        encoding = tokenizer(
            texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

    for prob in probs:
        pred_labels = [id2label[i] for i, p in enumerate(prob) if p > threshold]
        results.append({"labels": pred_labels, "probs": prob})
    return results

# ===== 使用示例 =====
if __name__ == "__main__":
    texts = [
        "The attacker exploited a vulnerability to achieve remote code execution.",
        "Malware established a reverse shell connection to the attacker server.",
        "An adversary attempted brute force on RDP to gain access."
    ]
    model, tokenizer, id2label = init_text_model(save_path, model_path=os.path.join(save_path, "best_model_state.pt"))

    results = predict_texts(texts, model, tokenizer, id2label)

