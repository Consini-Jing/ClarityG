import re
import base64
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel

MODEL_PATH = "/experiment/models/codebert-base"
SAVE_PATH  = "/experiment/models/best_cmdner_model.pt"

# ─────────────────────────────────────────
# 标签
# ─────────────────────────────────────────
LABEL2ID = {
    "O": 0,
    "B-Process": 1, "I-Process": 2,
    "B-File": 3,    "I-File": 4,
    "B-Socket": 5,  "I-Socket": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
def normalize_cmdline(cmd: str) -> str:
    cmd = re.sub(r'"{2,}', '', cmd)
    cmd = re.sub(r"'{2,}", '', cmd)

    enc_match = re.search(
        r'-En(?:c(?:o(?:d(?:e(?:d(?:C(?:o(?:m(?:m(?:a(?:nd?)?)?)?)?)?)?)?)?)?)?)?\s+([A-Za-z0-9+/=]{20,})',
        cmd, re.IGNORECASE
    )
    if enc_match:
        try:
            decoded = base64.b64decode(enc_match.group(1)).decode('utf-16-le')
            cmd = cmd[:enc_match.start()] + decoded
        except Exception:
            pass

    return cmd.strip()

class CmdNameAwareNER(nn.Module):
    def __init__(self, num_labels: int, model_path: str):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_path)
        d = self.encoder.config.hidden_size
        self.classifier = nn.Linear(d * 2, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, cmd_token_pos):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # (B, L, d)

        B, L, _ = H.shape

        h_cmd = H[torch.arange(B, device=H.device), cmd_token_pos]  # (B, d)
        h_cmd_exp = h_cmd.unsqueeze(1).expand(-1, L, -1)

        H_fuse = self.dropout(torch.cat([H, h_cmd_exp], dim=-1))
        logits = self.classifier(H_fuse)

        return logits


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizerFast.from_pretrained(
        MODEL_PATH,
        add_prefix_space=True
    )

    model = CmdNameAwareNER(NUM_LABELS, MODEL_PATH)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict(cmd: str, model, tokenizer, device, max_length=128):

    cmd = normalize_cmdline(cmd)

    tokens = cmd.split()
    if not tokens:
        return []

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids(batch_index=0)

    cmd_token_pos = next(
        (i for i, wid in enumerate(word_ids) if wid == 0), 0
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    cmd_pos_tensor = torch.tensor([cmd_token_pos], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, cmd_pos_tensor)

    preds = torch.argmax(logits, dim=-1)[0].tolist()

    word_preds = {}
    for i, wid in enumerate(word_ids):
        if wid is not None and wid not in word_preds:
            word_preds[wid] = ID2LABEL[preds[i]]

    spans = []
    current = None

    for i, token in enumerate(tokens):
        label = word_preds.get(i, "O")

        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = {
                "text": token,
                "type": label[2:],
                "start": i,
                "end": i
            }

        elif label.startswith("I-") and current:
            current["text"] += " " + token
            current["end"] = i

        else:
            if current:
                spans.append(current)
            current = None

    if current:
        spans.append(current)

    return spans


if __name__ == "__main__":
    model, tokenizer, device = load_model()
    while True:
        cmd = input("").strip()
        if cmd.lower() in ["exit", "quit"]:
            break
        spans = predict(cmd, model, tokenizer, device)
        print("\nresult：")
        for s in spans:
            print(f"  {s}")

        print("-" * 40)