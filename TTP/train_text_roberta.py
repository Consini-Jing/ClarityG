import os
import json
import torch
import gc
import re
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
from transformers import logging
logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

def clean_text(text):

    text = re.sub(r"\s+", " ", text)
    return text

class TextDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length, label2id):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        labels = item.get("labels", [])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        label_vec = torch.zeros(len(self.label2id))
        for l in labels:
            if l in self.label2id:
                label_vec[self.label2id[l]] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vec
        }

class RobertaClassifier(torch.nn.Module):
    def __init__(self, roberta, hidden_size, num_labels):
        super().__init__()
        self.roberta = roberta
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        return loss, logits

def evaluate(model, dataloader, device, threshold=0.05, label2id=None,
             save_path=None, raw_data=None, dataset_indices=None):

    model.eval()
    all_labels, all_preds = [], []

    total_loss = 0
    id2label = {v: k for k, v in label2id.items()} if label2id else None
    results = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    subset_acc = accuracy_score(all_labels, all_preds)

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
    for i in range(len(all_labels)):
        y_true = all_labels[i]
        y_pred = all_preds[i]

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

    return {
        "loss": avg_loss,
        "subset_acc": subset_acc,
        "micro_p": micro_p, "micro_r": micro_r, "micro_f05": micro_f05,
        "macro_p": macro_p, "macro_r": macro_r, "macro_f05": macro_f05,
        "sample_p": sample_p, "sample_r": sample_r, "sample_f05": sample_f05,
    }


if __name__ == "__main__":

    roberta_path = "./models/roberta-base"
    dataset_path = "./data/text/TTPs_text_filtered.json"
    save_path = "./text_roberta_0.5"
    max_length = 128
    batch_size = 64
    lr_encoder = 2e-5
    lr_classifier = 4e-5
    EPOCHS = 40
    device = "cuda" if torch.cuda.is_available() else "cpu"


    tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    roberta_encoder = RobertaModel.from_pretrained(roberta_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data_all = json.load(f)
    raw_data = [item for item in raw_data_all if item.get("type") == "text" and item.get("labels")]

    for item in raw_data:
        text = clean_text(item.get("text", ""))
        item["text"] = text

 
    all_labels = sorted({l for item in raw_data for l in item.get("labels", [])})
    label2id = {label: idx for idx, label in enumerate(all_labels)}
   
    dataset = TextDataset(raw_data, tokenizer, max_length, label2id)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = RobertaClassifier(roberta_encoder, hidden_size=768, num_labels=len(label2id))
    model = model.to(device)

    optimizer = AdamW([
        {"params": model.roberta.parameters(), "lr": lr_encoder},
        {"params": model.classifier.parameters(), "lr": lr_classifier}
    ])

    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, "train_log.csv")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,"
            "val_loss,val_subset_acc,val_micro_p,val_micro_r,val_micro_f05,"
            "val_macro_p,val_macro_r,val_macro_f05,"
            "val_sample_p,val_sample_r,val_sample_f05,"
            "test_loss,test_subset_acc,test_micro_p,test_micro_r,test_micro_f05,"
            "test_macro_p,test_macro_r,test_macro_f05,"
            "test_sample_p,test_sample_r,test_sample_f05\n"
        )

    best_score = 0.0
    for epoch in range(0,EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_results_file = os.path.join(save_path, f"val_predictions_epoch_{epoch+1}.jsonl")
        val_metrics = evaluate(
            model, val_loader, device, threshold=0.5,
            label2id=label2id,
            raw_data=raw_data, dataset_indices=val_indices
        )

        test_results_file = os.path.join(save_path, f"test_predictions_epoch_{epoch+1}.jsonl")
        test_metrics = evaluate(
            model, test_loader, device, threshold=0.5,
            label2id=label2id,
            raw_data=raw_data, dataset_indices=test_indices
        )

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")
        print(f"  [Val]  Loss={val_metrics['loss']:.4f}, SubsetAcc={val_metrics['subset_acc']:.4f}, "
              f"MicroF0.5={val_metrics['micro_f05']:.4f}, MacroF0.5={val_metrics['macro_f05']:.4f}, "
              f"SampleF0.5={val_metrics['sample_f05']:.4f}")
        print(f"  [Test] Loss={test_metrics['loss']:.4f}, SubsetAcc={test_metrics['subset_acc']:.4f}, "
              f"MicroF0.5={test_metrics['micro_f05']:.4f}, MacroF0.5={test_metrics['macro_f05']:.4f}, "
              f"SampleF0.5={test_metrics['sample_f05']:.4f}")


        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},{avg_train_loss:.4f},"
                f"{val_metrics['loss']:.4f},{val_metrics['subset_acc']:.4f},"
                f"{val_metrics['micro_p']:.4f},{val_metrics['micro_r']:.4f},{val_metrics['micro_f05']:.4f},"
                f"{val_metrics['macro_p']:.4f},{val_metrics['macro_r']:.4f},{val_metrics['macro_f05']:.4f},"
                f"{val_metrics['sample_p']:.4f},{val_metrics['sample_r']:.4f},{val_metrics['sample_f05']:.4f},"
                f"{test_metrics['loss']:.4f},{test_metrics['subset_acc']:.4f},"
                f"{test_metrics['micro_p']:.4f},{test_metrics['micro_r']:.4f},{test_metrics['micro_f05']:.4f},"
                f"{test_metrics['macro_p']:.4f},{test_metrics['macro_r']:.4f},{test_metrics['macro_f05']:.4f},"
                f"{test_metrics['sample_p']:.4f},{test_metrics['sample_r']:.4f},{test_metrics['sample_f05']:.4f}\n"
            )
        current_score = val_metrics['macro_f05']
        if current_score > best_score:
            best_score = current_score
            torch.save(model, os.path.join(save_path, "best_model.pt"))
            torch.save(model.state_dict(), os.path.join(save_path, "best_model_state.pt"))
            tokenizer.save_pretrained(f"{save_path}/tokenizer")
            with open(os.path.join(save_path, "label2id.json"), "w", encoding="utf-8") as f_json:
                json.dump(label2id, f_json, ensure_ascii=False, indent=4)


        gc.collect()
        torch.cuda.empty_cache()
