import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import json
import torch.nn as nn

save_path = "ClarityG/TTP/result/dual_0.4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualEncoder(nn.Module):
    def __init__(self, roberta_model, codebert_model, hidden_size=768, num_labels=10):
        super().__init__()
        self.roberta = roberta_model
        self.codebert = codebert_model
        self.W_text = nn.Linear(hidden_size, hidden_size)
        self.W_cmd = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, text_ids, text_mask, cmd_ids, cmd_mask):
        f_text = self.roberta(input_ids=text_ids, attention_mask=text_mask).pooler_output
        f_cmd = self.codebert(input_ids=cmd_ids, attention_mask=cmd_mask).pooler_output
        alpha_text = torch.sigmoid(self.W_text(f_text))
        alpha_cmd = torch.sigmoid(self.W_cmd(f_cmd))
        f_fusion = alpha_text * f_text + alpha_cmd * f_cmd
        logits = self.classifier(f_fusion)
        return logits

def init_dual_model(save_path, model_path=None):
    with open(os.path.join(save_path, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    tokenizer_text = RobertaTokenizer.from_pretrained(f"{save_path}/tokenizer_text")
    tokenizer_cmd = RobertaTokenizer.from_pretrained(f"{save_path}/tokenizer_cmd")

    roberta_encoder = RobertaModel.from_pretrained(
        "ClarityG/TTP/models/roberta-base"
    )
    codebert_encoder = RobertaModel.from_pretrained(
        "/root/ClarityG/TTP/models/codebert-base"
    )

    model = DualEncoder(
        roberta_encoder,
        codebert_encoder,
        hidden_size=768,
        num_labels=len(label2id)
    )

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    return model, tokenizer_text, tokenizer_cmd, id2label

def predict_dual(
    samples,
    model,
    tokenizer_text,
    tokenizer_cmd,
    id2label,
    threshold=0.4
):
    texts = [s["text"] for s in samples]
    cmds = [s["cmd"] for s in samples]
    text_inputs = tokenizer_text(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    cmd_inputs = tokenizer_cmd(
        cmds,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    text_ids = text_inputs["input_ids"].to(device)
    text_mask = text_inputs["attention_mask"].to(device)
    cmd_ids = cmd_inputs["input_ids"].to(device)
    cmd_mask = cmd_inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(text_ids, text_mask, cmd_ids, cmd_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

    results = []
    for prob in probs:
        pred_labels = [
            id2label[i] for i, p in enumerate(prob) if p > threshold
        ]
        results.append({"labels": pred_labels, "probs": prob})

    return results

if __name__ == "__main__":

    model, tokenizer_text, tokenizer_cmd, id2label = init_dual_model(
        save_path,
        model_path=os.path.join(save_path, "best_model_state.pt")
    )

    results = predict_dual(
        samples,
        model,
        tokenizer_text,
        tokenizer_cmd,
        id2label
    )
