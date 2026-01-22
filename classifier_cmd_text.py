import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

# 配置
MODEL_PATH = "TTP/models/roberta-base"
DATASET_DIR = "./datasets"
BATCH_SIZE = 16
MAX_LENGTH = 128
EPOCHS = 3


# 自定义数据集类
class BalancedCommandDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path) as f:
            data = json.load(f)

        self.texts = [d["text"] for d in data]
        self.labels = [d["label"] for d in data]

        # 动态填充（更节省内存）
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)


# 改进的评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "cmd_precision" : precision_score(labels, preds, pos_label=1, zero_division=0),
        "cmd_recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "cmd_f1" : f1_score(labels, preds, pos_label=1, zero_division=0)
    }


def main():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

    train_dataset = BalancedCommandDataset(f"{DATASET_DIR}/train.json", tokenizer)
    val_dataset = BalancedCommandDataset(f"{DATASET_DIR}/val.json", tokenizer)
    test_dataset = BalancedCommandDataset(f"{DATASET_DIR}/test.json", tokenizer)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,  # 微调时使用更小的学习率
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="cmd_recall",  # 优先优化命令召回率
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none"  # 禁用wandb等记录
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("开始微调...")
    trainer.train()
    trainer.save_model("./command-detection-roberta-balanced")
    tokenizer.save_pretrained("./command-detection-roberta-balanced")

    print("\n测试集评估结果:")
    test_results = trainer.evaluate(test_dataset)
    print(f"准确率: {test_results['eval_accuracy']:.4f}")
    print(f"命令类精确率 (Cmd Precision): {test_results['eval_cmd_precision']:.4f}")
    print(f"命令类召回率 (Cmd Recall): {test_results['eval_cmd_recall']:.4f}")
    print(f"命令类F1 (Cmd F1): {test_results['eval_cmd_f1']:.4f}")


if __name__ == "__main__":
    main()
