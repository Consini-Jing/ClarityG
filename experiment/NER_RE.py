import json
import re
import networkx as nx
from collections import defaultdict
from test import predict_single
from ner_eval.ner_inference import load_model,predict
model, tokenizer, device = load_model()

def strip_multiple_edge_quotes(s: str) -> str:

    if not s:
        return s
    s = s.strip()
    s = s.lstrip('"\'')
    s = s.rstrip('"\'')
    return s
def read_commands_from_file(file_path):

    commands = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    commands.append(stripped_line)
    except FileNotFoundError:
        return None
    except Exception as e:
        return None
    return commands
def load_patterns(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
def generate_tagged_sentences(entities, valid_types={'process', 'file', 'socket'}):
    cmd = []
    pairs=[]
    sentence=""
    for i in range(len(entities)):
        e1_text, e1_type = entities[i]
        if e1_type =="other":
            continue
        for j in range(len(entities)):
            e2_text, e2_type = entities[j]
            if e2_type == "other":
                continue
            if i < j and e2_type in valid_types and e1_text!=e2_text :
                pairs.append([i,j])
    for i in range(len(pairs)):
        index1,index2=pairs[i]
        for j in range(len(entities)):
            if j!=0:
                sentence+=" "
            e_text, e_type = entities[j]
            if j==index1:
                sentence=sentence+"<e1>"+e_text+"</e1>"
            elif j==index2:
                sentence = sentence + "<e2>" + e_text + "</e2>"
            else:
                sentence = sentence + e_text
        cmd.append(sentence)
        # print(sentence)
        sentence=""
    return cmd
def extract_entities_from_command(command):

    spans = predict(command, model, tokenizer, device)

    span_map = {}
    for span in spans:
        for token in span["text"].split():
            span_map[token] = span["type"].lower()

    # 完全对齐原来格式
    entities = []
    tokens = command.split()
    for token in tokens:
        matched_type = span_map.get(token)
        if matched_type:
            entities.append((token, matched_type))
        else:
            entities.append((token, "other"))

    return entities

def process_command_line_single(command):

    entities = extract_entities_from_command(command)

    entities_list=[[t, tp] for t, tp in entities]
    tagged_cmd = generate_tagged_sentences(entities_list)
    model_path="/root/ClarityG/R-BERT-master/model"
    corrected_entities = {}
    relation_entity_types = {
        'process-file-read(e1,e2)': ('process', 'file'),
        'process-file-write(e1,e2)': ('process', 'file'),
        'process-file-exec(e1,e2)': ('process', 'file'),
        'process-file-chmod(e1,e2)': ('process', 'file'),
        'process-file-unlink(e1,e2)': ('process', 'file'),
        'process-socket-send(e1,e2)': ('process', 'socket'),
        'process-socket-receive(e1,e2)': ('process', 'socket'),
        'process-process-fork(e1,e2)': ('process', 'process'),
        'process-process-inject(e1,e2)': ('process', 'process'),
        'process-process-unlink(e1,e2)': ('process', 'process')
    }
    for cmd in tagged_cmd:
        prediction,probs = predict_single(cmd, model_path)
        pred_idx = probs.index(max(probs))
        confidence = probs[pred_idx]

        if prediction!="Other" and prediction!="other" and confidence>=0.8:
            expected_e1, expected_e2 = relation_entity_types[prediction]
            e1 = re.search(r"<e1>(.*?)</e1>", cmd).group(1)
            e2 = re.search(r"<e2>(.*?)</e2>", cmd).group(1)
            # print(e1,e2)
            if(e1 not in corrected_entities):
                corrected_entities[e1]=expected_e1
            if (e2 not in corrected_entities):
                corrected_entities[e2] = expected_e2

    for i, (t, tp) in enumerate(entities_list):
        if t in corrected_entities:
            corrected_type = corrected_entities[t]
            if tp != corrected_type:
                entities_list[i][1] = corrected_type
    final_entities = [(t, tp) for t, tp in entities_list]
    return final_entities,tagged_cmd
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

ENTITY_TYPES = {"process", "file", "socket"}
def compute_entity_metrics_by_type(details):
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for item in details:
        for text, typ in item["TP"]:
            if typ in ENTITY_TYPES:
                stats[typ]["TP"] += 1

        for text, typ in item["FP"]:
            if typ in ENTITY_TYPES:
                stats[typ]["FP"] += 1

        for text, typ in item["FN"]:
            if typ in ENTITY_TYPES:
                stats[typ]["FN"] += 1

    results = {}
    for typ in ENTITY_TYPES:
        TP = stats[typ]["TP"]
        FP = stats[typ]["FP"]
        FN = stats[typ]["FN"]

        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        results[typ] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return results

if __name__ == "__main__":

    ground_truth_file = "/root/ClarityG/datasets/NER_bio1.txt"
    cmd_file = "/root/ClarityG/datasets/NER_cmd.txt"
    ground_truth = load_ground_truth_detailed(ground_truth_file)

