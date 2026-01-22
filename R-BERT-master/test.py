import os
import torch
from model import RBERT
from utils import load_tokenizer, get_label
import torch.nn.functional as F

def load_trained_model(model_dir, no_cuda=False):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    args = torch.load(os.path.join(model_dir, "training_args.bin"))
    label_list=['Other', 'process-file-read(e1,e2)', 'process-file-write(e1,e2)', 'process-file-exec(e1,e2)', 'process-file-chmod(e1,e2)', 'process-file-unlink(e1,e2)', 'process-socket-send(e1,e2)', 'process-socket-receive(e1,e2)', 'process-process-fork(e1,e2)', 'process-process-inject(e1,e2)', 'process-process-unlink(e1,e2)']
    tokenizer = load_tokenizer(args)

    model = RBERT.from_pretrained(model_dir, args=args)
    model.to(device)
    model.eval()

    return model, tokenizer, args, device, label_list


def prepare_single_input(text, tokenizer, args):
    cls_token_segment_id = 0
    pad_token_segment_id = 0
    sequence_a_segment_id = 0
    mask_padding_with_zero = True

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    tokens = tokenizer.tokenize(text)
    e11_p = tokens.index("<e1>")
    e12_p = tokens.index("</e1>")
    e21_p = tokens.index("<e2>")
    e22_p = tokens.index("</e2>")

    tokens[e11_p] = "$"
    tokens[e12_p] = "$"
    tokens[e21_p] = "#"
    tokens[e22_p] = "#"

    e11_p += 1
    e12_p += 1
    e21_p += 1
    e22_p += 1

    if args.add_sep_token:
        special_tokens_count = 2
    else:
        special_tokens_count = 1
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[: args.max_seq_len - special_tokens_count]

    if args.add_sep_token:
        tokens += [sep_token]
    token_type_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    token_type_ids = [cls_token_segment_id] + token_type_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = args.max_seq_len - len(input_ids)
    input_ids += [pad_token_id] * padding_length
    attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
    token_type_ids += [pad_token_segment_id] * padding_length

    e1_mask = [0] * args.max_seq_len
    e2_mask = [0] * args.max_seq_len
    for i in range(e11_p, e12_p + 1):
        e1_mask[i] = 1
    for i in range(e21_p, e22_p + 1):
        e2_mask[i] = 1

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([attention_mask], dtype=torch.long),
        torch.tensor([token_type_ids], dtype=torch.long),
        torch.tensor([e1_mask], dtype=torch.long),
        torch.tensor([e2_mask], dtype=torch.long),
    )


def predict_single(text, model_dir, no_cuda=False):
    model, tokenizer, args, device, label_list = load_trained_model(model_dir, no_cuda)
    input_ids, attention_mask, token_type_ids, e1_mask, e2_mask = prepare_single_input(text, tokenizer, args)

    with torch.no_grad():
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "token_type_ids": token_type_ids.to(device),
            "labels": None,
            "e1_mask": e1_mask.to(device),
            "e2_mask": e2_mask.to(device),
        }
        outputs = model(**inputs)
        logits = outputs[0]
        pred_idx = torch.argmax(logits, dim=1).item()
        probs = F.softmax(logits, dim=1)
        confidence = probs[0, pred_idx].item()
    return label_list[pred_idx],probs[0].tolist()


# 示例调用
if __name__ == "__main__":
    command_line = '<e1>powershell.exe</e1>  Set-ItemProperty -Force -Path  \'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\SecurityProviders\\WDigest\' -Name  \'<e2>UseLogonCredential</e2>\' -Value \'1\''
    model_path = "/root/ClarityG/R-BERT-master/model"
    prediction,confidence = predict_single(command_line, model_path)
