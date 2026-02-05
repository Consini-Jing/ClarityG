import os
import re
import gc
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
from tqdm import tqdm
import numpy as np

roberta_path = "./models/roberta-base"
codebert_path = "./models/codebert-base"
dataset_path = "./data/cmd/TTPs_cmd_text_filtered.json"
save_path = "./result/dual_0.4"
max_length = 128
batch_size = 64
EPOCHS = 50
lr_encoder = 2e-5
lr_classifier = 4e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").strip()
    return text

class TextCmdDataset(Dataset):
    def __init__(self, data, tokenizer_text, tokenizer_cmd, max_length, label2id):
        self.data = data
        self.tokenizer_text = tokenizer_text
        self.tokenizer_cmd = tokenizer_cmd
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        cmd = item.get("command", "")
        labels = item.get("labels", [])

        label_vec = torch.zeros(len(self.label2id), dtype=torch.float)
        for l in labels:
            if l in self.label2id:
                label_vec[self.label2id[l]] = 1.0

        text_inputs = self.tokenizer_text(
            text if text else "[PAD]",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        cmd_inputs = self.tokenizer_cmd(
            cmd if cmd else "[PAD]",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "text_input_ids": text_inputs["input_ids"].squeeze(0),
            "text_attention_mask": text_inputs["attention_mask"].squeeze(0),
            "cmd_input_ids": cmd_inputs["input_ids"].squeeze(0),
            "cmd_attention_mask": cmd_inputs["attention_mask"].squeeze(0),
            "labels": label_vec
        }

class DualEncoder(nn.Module):
    def __init__(self, roberta_model, codebert_model, hidden_size=768, num_labels=50):
        super().__init__()
        self.roberta = roberta_model
        self.codebert = codebert_model
        self.roberta.gradient_checkpointing_enable()
        self.codebert.gradient_checkpointing_enable()

        self.W_text = nn.Linear(hidden_size, hidden_size)
        self.W_cmd = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, text_ids, text_mask, cmd_ids, cmd_mask, labels=None):
        batch_size = text_ids.size(0)

        f_text = self.roberta(input_ids=text_ids, attention_mask=text_mask).pooler_output \
            if text_ids.sum() != 0 else torch.zeros(batch_size, self.roberta.config.hidden_size).to(text_ids.device)
        f_cmd = self.codebert(input_ids=cmd_ids, attention_mask=cmd_mask).pooler_output \
            if cmd_ids.sum() != 0 else torch.zeros(batch_size, self.codebert.config.hidden_size).to(cmd_ids.device)

        alpha_text = torch.sigmoid(self.W_text(f_text))
        alpha_cmd = torch.sigmoid(self.W_cmd(f_cmd))
        f_fusion = alpha_text * f_text + alpha_cmd * f_cmd

        logits = self.classifier(f_fusion)
        loss = nn.BCEWithLogitsLoss()(logits, labels) if labels is not None else None
        return loss, logits



def evaluate(model, dataloader, device, threshold=0.5, label2id=None,):
    model.eval()
    all_labels, all_preds = [], []
    subset_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            cmd_ids = batch["cmd_input_ids"].to(device)
            cmd_mask = batch["cmd_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(text_ids, text_mask, cmd_ids, cmd_mask, labels)
            total_loss += loss.item() if loss is not None else 0

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

            # Subset Accuracy
            for y_true, y_pred in zip(labels_np, preds):
                if set(np.where(y_true == 1)[0]).issubset(set(np.where(y_pred == 1)[0])):
                    subset_correct += 1
                total_samples += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    jaccard_per_sample = [jaccard_score(y_true, y_pred, average='binary')
                          for y_true, y_pred in zip(all_labels, all_preds)]
    jaccard_avg = np.mean(jaccard_per_sample)

    micro_p = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    micro_r = recall_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)

  
    def f_beta(p, r, beta=0.5):
        if p + r == 0:
            return 0.0
        return (1 + beta*beta) * p * r / (beta*beta * p + r)

    micro_f05 = f_beta(micro_p, micro_r, beta=0.5)
    macro_f05 = f_beta(macro_p, macro_r, beta=0.5)

    sample_ps, sample_rs, sample_fs = [], [], []
    for y_true, y_pred in zip(all_labels, all_preds):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f = f_beta(p, r, beta=0.5)

        sample_ps.append(p)
        sample_rs.append(r)
        sample_fs.append(f)

    sample_p = np.mean(sample_ps)
    sample_r = np.mean(sample_rs)
    sample_f05 = np.mean(sample_fs)

    avg_loss = total_loss / len(dataloader)
    subset_accuracy = subset_correct / total_samples

    return {
        "loss": avg_loss,
        "micro_p": micro_p, "micro_r": micro_r, "micro_f05": micro_f05,
        "macro_p": macro_p, "macro_r": macro_r, "macro_f05": macro_f05,
        "sample_p": sample_p, "sample_r": sample_r, "sample_f05": sample_f05,
        "jaccard": jaccard_avg,
        "subset_accuracy": subset_accuracy
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    tokenizer_text = RobertaTokenizer.from_pretrained(roberta_path)
    tokenizer_cmd = RobertaTokenizer.from_pretrained(codebert_path)

    roberta_encoder = RobertaModel.from_pretrained(roberta_path)
    codebert_encoder = RobertaModel.from_pretrained(codebert_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    for item in raw_data:
        item["text"] = clean_text(item.get("text", ""))
        item["command"] = clean_text(item.get("command", ""))



    all_labels = sorted({l for item in raw_data for l in item.get("labels", [])})
    label2id = {label: idx for idx, label in enumerate(all_labels)}
  

    dataset = TextCmdDataset(raw_data, tokenizer_text, tokenizer_cmd, max_length, label2id)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = DualEncoder(roberta_encoder, codebert_encoder, hidden_size=768, num_labels=len(label2id)).to(device)
    optimizer = AdamW([
        {"params": model.roberta.parameters(), "lr": lr_encoder},
        {"params": model.codebert.parameters(), "lr": lr_encoder},
        {"params": model.classifier.parameters(), "lr": lr_classifier},
        {"params": model.W_text.parameters(), "lr": lr_classifier},
        {"params": model.W_cmd.parameters(), "lr": lr_classifier}
    ])

    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, "train_log.csv")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,"
            "val_loss,val_subset_acc,val_micro_p,val_micro_r,val_micro_f05,"
            "val_macro_p,val_macro_r,val_macro_f05,"
            "val_sample_p,val_sample_r,val_sample_f05,"
            "val_jaccard,"
            "test_loss,test_subset_acc,test_micro_p,test_micro_r,test_micro_f05,"
            "test_macro_p,test_macro_r,test_macro_f05,"
            "test_sample_p,test_sample_r,test_sample_f05,"
            "test_jaccard\n"
        )

    best_score = 0.0
    for epoch in range(0,EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            cmd_ids = batch["cmd_input_ids"].to(device)
            cmd_mask = batch["cmd_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(text_ids, text_mask, cmd_ids, cmd_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_results_file = os.path.join(save_path, f"val_predictions_epoch_{epoch + 1}.jsonl")
        val_metrics = evaluate(model, val_loader, device, threshold=0.5, label2id=label2id)
        test_results_file = os.path.join(save_path, f"test_predictions_epoch_{epoch + 1}.jsonl")
        test_metrics = evaluate(model, test_loader, device, threshold=0.5, label2id=label2id)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")
        print(f"  [Val]  Loss={val_metrics['loss']:.4f}, SubsetAcc={val_metrics['subset_accuracy']:.4f}, "
              f"MicroF0.5={val_metrics['micro_f05']:.4f}, MacroF0.5={val_metrics['macro_f05']:.4f}, "
              f"SampleF0.5={val_metrics['sample_f05']:.4f}, Jaccard={val_metrics['jaccard']:.4f}")

        print(f"  [Test] Loss={test_metrics['loss']:.4f}, SubsetAcc={test_metrics['subset_accuracy']:.4f}, "
              f"MicroF0.5={test_metrics['micro_f05']:.4f}, MacroF0.5={test_metrics['macro_f05']:.4f}, "
              f"SampleF0.5={test_metrics['sample_f05']:.4f}, Jaccard={test_metrics['jaccard']:.4f}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1},{avg_train_loss:.4f},"
                f"{val_metrics['loss']:.4f},{val_metrics['subset_accuracy']:.4f},"
                f"{val_metrics['micro_p']:.4f},{val_metrics['micro_r']:.4f},{val_metrics['micro_f05']:.4f},"
                f"{val_metrics['macro_p']:.4f},{val_metrics['macro_r']:.4f},{val_metrics['macro_f05']:.4f},"
                f"{val_metrics['sample_p']:.4f},{val_metrics['sample_r']:.4f},{val_metrics['sample_f05']:.4f},"
                f"{val_metrics['jaccard']:.4f},"
                f"{test_metrics['loss']:.4f},{test_metrics['subset_accuracy']:.4f},"
                f"{test_metrics['micro_p']:.4f},{test_metrics['micro_r']:.4f},{test_metrics['micro_f05']:.4f},"
                f"{test_metrics['macro_p']:.4f},{test_metrics['macro_r']:.4f},{test_metrics['macro_f05']:.4f},"
                f"{test_metrics['sample_p']:.4f},{test_metrics['sample_r']:.4f},{test_metrics['sample_f05']:.4f},"
                f"{test_metrics['jaccard']:.4f}\n"
            )

        current_score = val_metrics['macro_f05']
        if current_score > best_score:
            best_score = current_score
            torch.save(model, os.path.join(save_path, "best_model.pt"))
            torch.save(model.state_dict(), os.path.join(save_path, "best_model_state.pt"))
            tokenizer_text.save_pretrained(f"{save_path}/tokenizer_text")
            tokenizer_cmd.save_pretrained(f"{save_path}/tokenizer_cmd")
            with open(os.path.join(save_path, "label2id.json"), "w", encoding="utf-8") as f_json:
                json.dump(label2id, f_json, ensure_ascii=False, indent=4)


        gc.collect()
        torch.cuda.empty_cache()
