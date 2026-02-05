# check_nan_samples.py
import torch
import torch.nn as nn
from transformers import BertTokenizer
import pandas as pd
from model import RBERT  
from types import SimpleNamespace


args = SimpleNamespace()
args.dropout_rate = 0.1  
args.max_seq_len = 256   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = "bert-base-uncased"  
MAX_SEQ_LEN = 256
TRAIN_FILE = "./data_CTI/train.tsv"


LABEL_LIST = [
    "Other",
    "process-file-read(e1,e2)",
    "process-file-write(e1,e2)",
    "process-file-exec(e1,e2)",
    "process-file-chmod(e1,e2)",
    "process-file-unlink(e1,e2)",
    "process-socket-send(e1,e2)",
    "process-socket-receive(e1,e2)",
    "process-process-fork(e1,e2)",
    "process-process-inject(e1,e2)",
    "process-process-unlink(e1,e2)"
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}


def preprocess_text(text, tokenizer, max_len):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = RBERT.from_pretrained(MODEL_NAME, args=args, num_labels=len(LABEL_LIST))

    model.to(DEVICE)
    model.eval()

    df = pd.read_csv(TRAIN_FILE, sep="\t", header=None, names=["label", "text"])

    nan_found = False

    for idx, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        input_ids, attention_mask = preprocess_text(text, tokenizer, MAX_SEQ_LEN)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        if torch.isnan(logits).any():
            nan_found = True
            print("=== NaN LOSS detected ===")
            print(f"Index: {idx}")
            print(f"Text: {text}")
            print(f"Label: {label}")
            print(f"Logits: {logits}\n")

    if not nan_found:
        print("No NaN samples found.")


if __name__ == "__main__":
    main()
