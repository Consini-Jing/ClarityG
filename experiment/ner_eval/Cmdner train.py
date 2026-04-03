import re
import base64
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import os

DATA_PATH   = "../../datasets/NER_bio.txt"
MODEL_PATH  = "../../experiment/models/codebert-base"
SAVE_PATH   = "../../experiment/models/best_cmdner_model.pt"
RESULT_PATH = "results/NER_eval_result.json"
MAX_LEN     = 128
BATCH_SIZE  = 8
EPOCHS      = 20
LR          = 2e-5

LABEL2ID = {
    "O": 0,
    "B-Process": 1, "I-Process": 2,
    "B-File": 3,    "I-File": 4,
    "B-Socket": 5,  "I-Socket": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def load_bio_file(filepath: str):
    samples = []
    tokens, labels = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.strip() == "":
                if tokens:
                    samples.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            label = parts[-1]
            token = " ".join(parts[:-1])

            if label not in LABEL2ID:
                continue

            tokens.append(token)
            labels.append(label)

    if tokens:
        samples.append({"tokens": tokens, "labels": labels})

    return samples


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
class CmdlineNERDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = list(sample["tokens"])
        labels = list(sample["labels"])

        if tokens:
            tokens[0] = normalize_cmdline(tokens[0]) or tokens[0]

        filtered = [(t, l) for t, l in zip(tokens, labels) if t.strip()]
        if not filtered:
            filtered = [("<empty>", "O")]
        tokens, labels = zip(*filtered)
        tokens, labels = list(tokens), list(labels)

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)

        aligned_labels = []
        prev_word_id   = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(LABEL2ID[labels[word_id]])
            else:
                aligned_labels.append(-100)
            prev_word_id = word_id

        cmd_token_pos = next(
            (i for i, wid in enumerate(word_ids) if wid == 0), 0
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
            "cmd_token_pos":  torch.tensor(cmd_token_pos,  dtype=torch.long),
        }

class CmdNameAwareNER(nn.Module):
    def __init__(self, num_labels: int, model_path: str):
        super().__init__()
        self.encoder    = RobertaModel.from_pretrained(model_path)
        d               = self.encoder.config.hidden_size
        self.classifier = nn.Linear(d * 2, num_labels)
        self.dropout    = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, cmd_token_pos, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state
        B, L, _ = H.shape

        h_cmd     = H[torch.arange(B, device=H.device), cmd_token_pos]
        h_cmd_exp = h_cmd.unsqueeze(1).expand(-1, L, -1)
        H_fuse    = self.dropout(torch.cat([H, h_cmd_exp], dim=-1))

        logits = self.classifier(H_fuse)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, NUM_LABELS),
                labels.view(-1)
            )
        return loss, logits

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        cmd_token_pos  = batch["cmd_token_pos"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, cmd_token_pos, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, desc, save_path=None):

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            cmd_token_pos  = batch["cmd_token_pos"].to(device)

            _, logits = model(input_ids, attention_mask, cmd_token_pos)
            preds     = torch.argmax(logits, dim=-1)

            for pred_seq, label_seq in zip(preds, labels):
                pred_tags, true_tags = [], []
                for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l == -100:
                        continue
                    pred_tags.append(ID2LABEL[p])
                    true_tags.append(ID2LABEL[l])
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    report_str  = classification_report(all_labels, all_preds)

    overall_p  = precision_score(all_labels, all_preds)
    overall_r  = recall_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds)

    print(f"\n{'='*55}")
    print(f"{'='*55}")
    print(report_str)
    print(f"  Overall  P: {overall_p:.4f}  R: {overall_r:.4f}  F1: {overall_f1:.4f}")
    print(f"{'='*55}\n")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        per_class = {}
        for key, metrics in report_dict.items():
            if not isinstance(metrics, dict):
                continue
            if key in ("micro avg", "macro avg", "weighted avg"):
                continue
            per_class[key] = {
                "precision": round(metrics["precision"], 4),
                "recall":    round(metrics["recall"],    4),
                "f1":        round(metrics["f1-score"],  4),
                "support":   int(metrics["support"]),
            }

        result = {
            "desc": desc,
            "overall": {
                "precision": round(overall_p,  4),
                "recall":    round(overall_r,  4),
                "f1":        round(overall_f1, 4),
            },
            "per_class":  per_class,
            "report_str": report_str,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


    return overall_f1


def predict(raw_cmd: str, model, tokenizer, device, max_length=128):

    cmd    = normalize_cmdline(raw_cmd)
    tokens = cmd.split()
    if not tokens:
        return []

    model.eval()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids      = encoding.word_ids(batch_index=0)
    cmd_token_pos = next(
        (i for i, wid in enumerate(word_ids) if wid == 0), 0
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    cmd_pos_tensor = torch.tensor([cmd_token_pos], dtype=torch.long).to(device)

    with torch.no_grad():
        _, logits = model(input_ids, attention_mask, cmd_pos_tensor)
    preds = torch.argmax(logits, dim=-1)[0].tolist()

    word_preds = {}
    for i, wid in enumerate(word_ids):
        if wid is not None and wid not in word_preds:
            word_preds[wid] = ID2LABEL[preds[i]]

    spans, current = [], None
    for i, token in enumerate(tokens):
        label = word_preds.get(i, "O")
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = {"text": token, "type": label[2:], "start": i, "end": i}
        elif label.startswith("I-") and current:
            current["text"] += " " + token
            current["end"]   = i
        else:
            if current:
                spans.append(current)
            current = None
    if current:
        spans.append(current)

    return spans


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    all_samples = load_bio_file(DATA_PATH)

    train_val, test_samples = train_test_split(
        all_samples, test_size=0.1, random_state=42
    )
    train_samples, val_samples = train_test_split(
        train_val, test_size=0.111, random_state=42
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH, add_prefix_space=True)

    train_loader = DataLoader(
        CmdlineNERDataset(train_samples, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        CmdlineNERDataset(val_samples, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        CmdlineNERDataset(test_samples, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE
    )

    model = CmdNameAwareNER(NUM_LABELS, MODEL_PATH).to(device)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_f1, patience, patience_counter = 0.0, 5, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f}")

        val_f1 = evaluate(model, val_loader, device, desc=f"Epoch{epoch}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    evaluate(
        model, test_loader, device,
        desc="test",
        save_path=RESULT_PATH
    )

