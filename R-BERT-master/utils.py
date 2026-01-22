import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_fscore_support, f1_score
# from official_eval import official_f1

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1_all(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1_all(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)            # 你原来的
    f1 = f1_score(labels, preds, average=average)   # 你原来的

    # 新增：precision / recall
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0
    )
    return {
        "acc": acc,
        "precision": precision,   # 新增
        "recall": recall,         # 新增
        "f1": f1,
    }
def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(labels, preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }
