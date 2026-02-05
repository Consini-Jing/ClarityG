#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json


def load_ground_truth_detailed(file_path):

    ground_truth = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_cmd_tokens = []
    current_labels = []

    def merge_spans(tokens, labels):
   
        merged = []
        span_tokens = []
        span_type = None  

        def flush_span():
            if span_tokens:
                merged.append((" ".join(span_tokens), span_type))

        for tok, lab in zip(tokens, labels):
            if lab == "O":
                lab_type = "other"
                lab_prefix = "O"
            else:
                lab_prefix = lab[0]  
                lab_type = lab[2:].lower()

            if lab_prefix == "B":
              
                flush_span()
                span_tokens = [tok]
                span_type = lab_type
            elif lab_prefix == "I":
               
                if span_tokens and span_type == lab_type and span_type != "other":
                    span_tokens.append(tok)
                else:
                   
                    flush_span()
                    span_tokens = [tok]
                    span_type = lab_type
            else:  
                flush_span()
                span_tokens = [tok]
                span_type = lab_type

        flush_span()
        return merged


    for line in lines:
        line = line.strip()

        if not line:
            if current_cmd_tokens:
                cmd_text = " ".join(current_cmd_tokens)
                merged = merge_spans(current_cmd_tokens, current_labels)
                ground_truth.append((cmd_text, merged))
                current_cmd_tokens = []
                current_labels = []
            continue

        parts = line.split()
        if len(parts) >= 2:
            current_cmd_tokens.append(parts[0])
            current_labels.append(parts[1])

    if current_cmd_tokens:
        cmd_text = " ".join(current_cmd_tokens)
        merged = merge_spans(current_cmd_tokens, current_labels)
        ground_truth.append((cmd_text, merged))

    return ground_truth


def load_special_phrases(special_file):

    special_phrases = {}
    try:
        with open(special_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    phrase, entity_type = line.split('\t', 1)
                    special_phrases[phrase] = entity_type

        return special_phrases
    except Exception as e:
        return {}

def match_entity(token, patterns):
    for major_category, subcategories in patterns.items():
        for subcategory, regex_list in subcategories.items():
            for regex in regex_list:
                try:
                    pattern = re.compile(regex, re.IGNORECASE)
                except re.error:
                    continue
                if pattern.search(token):
                    return major_category
    return None
def extract_entities_from_command(command, patterns, special_phrases):

    entities=[]
    used_positions = set()  

    sorted_special_phrases = sorted(special_phrases.items(), key=lambda x: len(x[0]), reverse=True)

    for phrase, entity_type in sorted_special_phrases:
        start = 0
        while True:
            pos = command.find(phrase, start)
            if pos == -1:
                break

            left_ok = (pos == 0) or (command[pos - 1] == ' ')
            right_ok = (pos + len(phrase) == len(command)) or (command[pos + len(phrase)] == ' ')

            if left_ok and right_ok:
                overlap = False
                for used_start, used_end in used_positions:
                    if not (pos >= used_end or pos + len(phrase) <= used_start):
                        overlap = True
                        break

                if not overlap:
                    used_positions.add((pos, pos + len(phrase)))
                    entities.append((phrase, entity_type.lower()))

            start = pos + 1


    if used_positions:
        sorted_positions = sorted(used_positions)
        segments = []
        last_end = 0

        for start, end in sorted_positions:
            if last_end < start:
                segments.append(command[last_end:start])
            last_end = end

        if last_end < len(command):
            segments.append(command[last_end:])

        for segment in segments:
            if segment.strip():
                tokens = segment.split()
                for token in tokens:
                    category = match_entity(token, patterns)
                    if category and category != "other":
                        entities.append((token, category))
                    else:
                        entities.append((token,"other"))
    else:
        tokens = command.split()
        for token in tokens:
            category = match_entity(token, patterns)
            if category and category != "other":
                entities.append((token, category))
            else:
                entities.append((token, "other"))

    entities.sort(key=lambda x: command.find(x[0]))
    return entities


def process_cmd_file(cmd_file, regex_file, special_file):

    with open(regex_file, 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    special_phrases = load_special_phrases(special_file)

    try:
        with open(cmd_file, 'r', encoding='utf-8') as f:
            commands = [line.strip() for line in f if line.strip()]

    except Exception as e:
      
        return []

    results = []

    for i, command in enumerate(commands):
        entities = extract_entities_from_command(command, patterns, special_phrases)
        results.append((command, entities))


    return results

def evaluate_with_details(ground_truth, regex_results, target_types={"process","file","socket"}):

    total_tp_all = total_fp_all = total_fn_all = 0
    total_tp_target = total_fp_target = total_fn_target = 0
    total_gt_all = 0
    total_gt_target = 0
    detailed_results = []
    all_errors = []


    N = min(len(ground_truth), len(regex_results))

    for idx in range(N):
        gt_cmd, gt_spans = ground_truth[idx]
        pred_cmd, pred_spans = regex_results[idx]

        gt_set_all = set(gt_spans)
        pred_set_all = set(pred_spans)

        tp_set_all = gt_set_all & pred_set_all
        fp_set_all = pred_set_all - gt_set_all
        fn_set_all = gt_set_all - pred_set_all

        total_tp_all += len(tp_set_all)
        total_fp_all += len(fp_set_all)
        total_fn_all += len(fn_set_all)
        total_gt_all += len(gt_set_all)

        gt_set_target = set([x for x in gt_spans if x[1] in target_types])
        pred_set_target = set([x for x in pred_spans if x[1] in target_types])

        tp_set_target = gt_set_target & pred_set_target
        fp_set_target = pred_set_target - gt_set_target
        fn_set_target = gt_set_target - pred_set_target

        total_tp_target += len(tp_set_target)
        total_fp_target += len(fp_set_target)
        total_fn_target += len(fn_set_target)
        total_gt_target += len(gt_set_target)

        detailed_results.append({
            "index": idx,
            "cmd": gt_cmd,
            "TP": list(tp_set_all),
            "FP": list(fp_set_all),
            "FN": list(fn_set_all)
        })

  
        if fp_set_all or fn_set_all:
            all_errors.append({
                "index": idx,
                "cmd": gt_cmd,
                "gt": gt_set_all,
                "pred": pred_set_all,
                "FP": list(fp_set_all),
                "FN": list(fn_set_all)
            })

    precision_all = total_tp_all / (total_tp_all + total_fp_all + 1e-10)
    recall_all = total_tp_all / (total_tp_all + total_fn_all + 1e-10)
    f1_all = 2 * precision_all * recall_all / (precision_all + recall_all + 1e-10)
    accuracy_all = total_tp_all / (total_gt_all + 1e-10)

    precision_target = total_tp_target / (total_tp_target + total_fp_target + 1e-10)
    recall_target = total_tp_target / (total_tp_target + total_fn_target + 1e-10)
    f1_target = 2 * precision_target * recall_target / (precision_target + recall_target + 1e-10)
    accuracy_target = total_tp_target / (total_gt_target + 1e-10)



    return (precision_all, recall_all, accuracy_all, f1_all,
            precision_target, recall_target, accuracy_target, f1_target,
            detailed_results)

if __name__ == "__main__":
    ground_truth_file = "ClarityG/datasets/NER_bio.txt"
    cmd_file = "ClarityG/datasets/NER_cmd.txt"
    regex_file = "ClarityG/regexPattern.json"
    special_file = "ClarityG/datasets/NER_teshu.txt"

    ground_truth=load_ground_truth_detailed(ground_truth_file)
    regex_results = process_cmd_file(cmd_file, regex_file, special_file)

    p, r, acc, f1, details = evaluate_with_details(ground_truth, regex_results)

    print("Precision:", p)
    print("Recall:", r)
    print("Accuracy:", acc)
    print("F1:", f1)



