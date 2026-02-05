from comRecognition import detect_commands_in_file
import numpy as np
from experiment.CAG import command_analysis_rule
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
import difflib
import graphviz
from itertools import islice
from pathlib import Path
import time
from TTP.inference_text import init_text_model,predict_texts
from TTP.inference_dual import init_dual_model,predict_dual
import json
import torch
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_report_and_graph(json_path, idx):
    with open(json_path, encoding="utf-8") as f:
        line = next(islice(f, idx, idx+1), None)
        if line is None:          
            return None, [], []
        data = json.loads(line.strip())

    doc_key   = data.get("doc_key")
    ners      = data.get("ners", [])
    relations = data.get("relations", [])
    return doc_key, ners, relations

def convert_report_relations(ners, relations, type_mapping):
    converted_relations = []

    for rel in relations:
        idx1, idx2, rel_type = rel
        node1 = ners[idx1]
        node2 = ners[idx2]

        text1, text2 = node1[-1], node2[-1]
        type1 = type_mapping.get(node1[2], "other")
        type2 = type_mapping.get(node2[2], "other")

        converted_relations.append([text1, type1, text2, type2, rel_type])

    return converted_relations

def deduplicate_relations(relations):
    seen = set()
    deduplicated = []

    for rel in relations:
        key = (rel[0], rel[1], rel[2], rel[3], rel[4])

        if key not in seen:
            seen.add(key)
            deduplicated.append(rel)

    return deduplicated

def normalize_report_nodes(ners, type_mapping):
    normalized = []
    for ner in ners:
        start, end, type_code, _, text = ner
        node_type = type_mapping.get(type_code, "other")
        normalized.append({
            "text": text,
            "type": node_type,
            "source": "report"
        })
    return normalized
def normalize_cmd_nodes(cmd_relations):

    normalized = []
    for rel in cmd_relations:
        e1, t1, e2, t2, _ = rel
        normalized.append({
            "text": e1,
            "type": t1,
            "source": "cmd"
        })
        normalized.append({
            "text": e2,
            "type": t2,
            "source": "cmd"
        })
    seen = set()
    unique = []
    for node in normalized:
        key = (node["text"], node["type"])
        if key not in seen:
            seen.add(key)
            unique.append(node)
    return unique

def load_encoder():
    model = SentenceTransformer('ClarityG/experiment/model/all-MiniLM-L6-v2')
    node_types = ["process", "file", "socket","other"]
    type_embeddings = {
        t: np.eye(len(node_types))[i]
        for i, t in enumerate(node_types)
    }
    return model, type_embeddings

def encode_node(model, type_embeddings, node_text, node_type):
    text_vec = model.encode([node_text])[0]
    text_vec = np.array(text_vec, dtype=np.float32).reshape(-1)
    type_vec = type_embeddings.get(
        node_type,
        type_embeddings.get("other", np.zeros(len(next(iter(type_embeddings.values())))))
    )
    type_vec = np.array(type_vec, dtype=np.float32).reshape(-1)
    return np.concatenate([text_vec, type_vec])

def encode_nodes(model, type_embeddings, nodes):
    encoded = []
    for node in nodes:
        vec = encode_node(model, type_embeddings, node["text"], node["type"])
        node["vec"] = vec
        encoded.append(vec)
    return np.vstack(encoded).astype(np.float32)

def build_faiss_index(vectors, dim):
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index
def safe_stack(vectors):
    clean_vecs = []
    for v in vectors:
        arr = np.array(v, dtype=np.float32).reshape(-1)
        clean_vecs.append(arr)
    return np.vstack(clean_vecs).astype('float32')
def search_candidates(report_nodes, cmd_nodes, top_k=5):
    report_vecs = safe_stack([n["vec"] for n in report_nodes])
    cmd_vecs = safe_stack([n["vec"] for n in cmd_nodes])
    dim = report_vecs.shape[1]

    index = build_faiss_index(report_vecs, dim)

    faiss.normalize_L2(cmd_vecs)
    scores, indices = index.search(cmd_vecs, top_k)

    results = []
    for i, cmd_node in enumerate(cmd_nodes):
        candidates = []
        for j, idx in enumerate(indices[i]):
            candidate = report_nodes[idx]
            sim = scores[i][j]
            candidates.append({
                "report_node": candidate,
                "similarity": float(sim)
            })
        results.append({
            "cmd_node": cmd_node,
            "candidates": candidates
        })
    return results

def normalize_relations(relations, relation_mapping):
    normalized = []
    for r in relations:
        e1, t1, e2, t2, rel = r
        norm_rel = relation_mapping.get(rel, "other") 
        normalized.append([e1, t1, e2, t2, norm_rel])
    return normalized
def normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r'\s+', '', text)  
    text = text.replace("\\", "/")   
    text = re.sub(r'[^a-z0-9:/._-]', '', text) 
    return text
def get_last_segment(path):
    return os.path.basename(path)
def enhanced_match(cmd_node, report_node, threshold=0.85):
    cmd_text, cmd_type = cmd_node["text"], cmd_node["type"]
    report_text, report_type = report_node["text"], report_node["type"]

    if cmd_type != report_type:
        return False, 0.0

    cmd_norm = normalize_text(cmd_text)
    report_norm = normalize_text(report_text)

    cmd_last = get_last_segment(cmd_norm)
    report_last = get_last_segment(report_norm)

    if cmd_last == report_last and cmd_last != "":
        return True, 1.0

    similarity = difflib.SequenceMatcher(None, cmd_last, report_last).ratio()
    if similarity >= threshold:
        return True, similarity

    return False, similarity
def match_nodes_with_fallback(report_nodes, cmd_nodes, semantic_threshold=0.9, enhanced_threshold=0.85, top_k=1):
    match_results = search_candidates(report_nodes, cmd_nodes, top_k)
    matched_nodes = []
    unmatched_cmd_nodes = []
    for item in match_results:
        cmd_node = item["cmd_node"]
        best_cand = max(item["candidates"], key=lambda x: x["similarity"])
        if best_cand["similarity"] >= semantic_threshold:
            rn = best_cand["report_node"]
            matched_nodes.append({
                "cmd_text": cmd_node["text"],
                "cmd_type": cmd_node["type"],
                "report_text": rn["text"],
                "report_type": rn["type"],
                "similarity": best_cand["similarity"]
            })
        else:
            unmatched_cmd_nodes.append(cmd_node)

    for cmd_node in unmatched_cmd_nodes:
        for rep_node in report_nodes:
            matched, score = enhanced_match(cmd_node, rep_node, threshold=enhanced_threshold)
            if matched:
                matched_nodes.append({
                    "cmd_text": cmd_node["text"],
                    "cmd_type": cmd_node["type"],
                    "report_text": rep_node["text"],
                    "report_type": rep_node["type"],
                    "similarity": score
                })
                break

    return matched_nodes
def merge_graphs(report_nodes, report_relations, cmd_nodes, cmd_relations, matched_nodes):

    match_map = {m["cmd_text"].lower(): m["report_text"] for m in matched_nodes}

    merged_nodes = {n["text"].lower(): n.copy() for n in report_nodes}
    merged_relations = report_relations.copy()

    for n in cmd_nodes:
        key = n["text"].lower()
        if key in match_map:
            rep_text = match_map[key]
            rep_key = rep_text.lower()

            if rep_key in merged_nodes:
           
                old_type = merged_nodes[rep_key]["type"]
                new_type = n["type"]

                if not isinstance(old_type, list):
                    old_type = [old_type]
                if isinstance(new_type, list):
                    combined_types = old_type + new_type
                else:
                    combined_types = old_type + [new_type]
                merged_nodes[rep_key]["type"] = list(set(combined_types))

                merged_nodes[rep_key].setdefault("aliases", []).append(n["text"])

                src = merged_nodes[rep_key].get("source", [])
                if isinstance(src, str):  
                    src = [src]
                if "cmd" not in src:
                    src.append("cmd")
                merged_nodes[rep_key]["source"] = src

                merged_nodes[rep_key]["color"] = "blue"
            else:
                merged_nodes[rep_key] = {
                    "text": rep_text,
                    "type": n["type"],
                    "source": ["report", "cmd"],
                    "aliases": [n["text"]],
                    "color": "blue",
                }
        else:
            merged_nodes[key] = {
                "text": n["text"],
                "type": n["type"],
                "source": ["cmd"],
                "aliases": [],
                "color": "red",
            }

    for e1, t1, e2, t2, rel in cmd_relations:
        k1, k2 = e1.lower(), e2.lower()
        if k1 in match_map:
            e1 = match_map[k1]
        if k2 in match_map:
            e2 = match_map[k2]
        merged_relations.append([e1, t1, e2, t2, rel])

    return list(merged_nodes.values()), merged_relations

def build_text_and_text_cmd(lines_with_index):
    text_only = []
    text_cmd_pairs = []

    last_text = None
    last_text_start = None

    for sent, start_idx, is_cmd in lines_with_index:
        if not is_cmd:
            text_only.append(sent)
            last_text = sent
        else:
            text_cmd_pairs.append({
                "text": last_text if last_text else "",
                "cmd": sent
            })

    return text_only, text_cmd_pairs


def build_sentence_entities_map(lines_with_index, ners):
    sentence_entities_map = []

    for i, (line_text, line_start, _) in enumerate(lines_with_index):
        if i < len(lines_with_index) - 1:
            line_end = lines_with_index[i + 1][1]
        else:
            line_end = max([ner[1] for ner in ners]) + 1

        entities_in_line = [ner for ner in ners if ner[0] >= line_start and ner[1] < line_end]
        sentence_entities_map.append(entities_in_line)

    return sentence_entities_map


def build_merged_attack_graph(models,report_path, cti_path,idx, regex_file,special_file,threshold=0.9, top_k=1):

    doc_key, ners, relations = load_report_and_graph(cti_path,idx)
    commands,lines_with_index=detect_commands_in_file(
        file_path=report_path,
        output_path="ClarityG/experiment/ag/command_results.txt",
        threshold=0.9
    )
    text_samples, cmd_samples = build_text_and_text_cmd(lines_with_index)
    text_ttp = predict_texts(text_samples, models["text_model"], models["tokenizer_text"], models["id2label_text"])
    cmd_ttp = predict_dual(cmd_samples, models["cmd_model"], models["tokenizer_text_cmd"], models["tokenizer_cmd"], models["id2label_cmd"])

    cmd_relations_all=[]
    for cmd in commands:
        _, entity_list, rule_relations = command_analysis_rule(cmd, regex_file, special_file)
        cmd_relations_all.extend(rule_relations)
    cmd_relations_all=deduplicate_relations(cmd_relations_all)

    type_mapping = {
        "SO": "socket",
        "TP": "process",
        "MP": "process",
        "SF": "file",
        "TF": "file",
        "MF": "file",
    }
    report_nodes = normalize_report_nodes(ners, type_mapping)
    for node in report_nodes:
        node_text = node['text']
        line_idx = next(
            (i for i, line in enumerate(lines_with_index)
             if not line[2] and node_text in line[0]),
            None
        )
        node['line_idx'] = line_idx
    cmd_nodes = normalize_cmd_nodes(cmd_relations_all)
    for node in cmd_nodes:
        node_text = node['text']
        line_idx = next(
            (i for i, line in enumerate(lines_with_index)
             if line[2] and node_text in line[0]),
            None
        )
        node['line_idx'] = line_idx

    all_nodes = report_nodes + cmd_nodes

    report_relations_converted = convert_report_relations(ners, relations, type_mapping)

    model, type_embeddings = load_encoder()
    encoded_nodes = encode_nodes(model, type_embeddings, all_nodes)

    for node, vec in zip(all_nodes, encoded_nodes):
        node["vec"] = vec

    text_list_raw = defaultdict(list)
    cmd_list_raw = defaultdict(list)
    for node in all_nodes:
        line_idx = node.get("line_idx")
        if line_idx is None:
            continue
        if node.get("source") == "cmd":
            cmd_list_raw[line_idx].append(node)
        else:
            text_list_raw[line_idx].append(node)
    text_list = defaultdict(list)
    text_line_mapping = {}
    for new_idx, old_line_idx in enumerate(sorted(text_list_raw.keys())):
        text_list[new_idx] = text_list_raw[old_line_idx]
        text_line_mapping[old_line_idx] = new_idx

    cmd_list = defaultdict(list)
    cmd_line_mapping = {}
    for new_idx, old_line_idx in enumerate(sorted(cmd_list_raw.keys())):
        cmd_list[new_idx] = cmd_list_raw[old_line_idx]
        cmd_line_mapping[old_line_idx] = new_idx

    report_nodes = [n for n in all_nodes if n["source"] == "report"]
    cmd_nodes = [n for n in all_nodes if n["source"] == "cmd"]

    matched_nodes = []
    for cmd_line_idx, cmd_nodes_in_line in cmd_list.items():
        print(cmd_line_idx)
        cmd_ttps = set(cmd_ttp[cmd_line_idx].get("labels", []))
        if not cmd_ttps:
            continue
        for text_line_idx, text_nodes_in_line in text_list.items():
            print(text_line_idx)
            text_ttps = set(text_ttp[text_line_idx].get("labels", []))
            if cmd_ttps & text_ttps:
                matched_nodes.extend(
                    match_nodes_with_fallback(text_nodes_in_line, cmd_nodes_in_line)
                )

    relation_mapping = {

        "RD": "read",
        "WR": "write",
        "EX": "exec",
        "CD": "chmod",
        "UK": "unlink",
 
        "ST": "send",
        "RF": "receive",

        "FR": "fork",
        "IJ": "inject",

        "process-file-read(e1,e2)": "read",
        "process-file-write(e1,e2)": "write",
        "process-file-exec(e1,e2)": "exec",
        "process-file-chmod(e1,e2)": "chmod",
        "process-file-unlink(e1,e2)": "unlink",
        "process-socket-send(e1,e2)": "send",
        "process-socket-receive(e1,e2)": "receive",
        "process-process-fork(e1,e2)": "fork",
        "process-process-inject(e1,e2)": "inject",
        "process-process-unlink(e1,e2)": "unlink",
    }
    report_relations_normalized = normalize_relations(report_relations_converted,relation_mapping)
    cmd_relations_normalized = normalize_relations(cmd_relations_all,relation_mapping)

    merged_nodes, merged_relations = merge_graphs(
        report_nodes,
        report_relations_normalized,
        cmd_nodes,
        cmd_relations_normalized,
        matched_nodes
    )
    return merged_nodes, merged_relations

node_shapes = {
    'file': 'rect',             
    'socket': 'diamond',         
    'process': 'ellipse'           
}
def draw_graph(nodes, relations, output_filename):

    graph = graphviz.Digraph(output_filename, filename=output_filename + ".dot")

    graph.attr(
        rankdir='LR',
        size='9',
        splines='true',
        nodesep='0.3',
        ranksep='0.5',
        fontsize='10',
        overlap='false',
        engine='dot'
    )

    added_nodes = {}

    def safe_label(text):
        return text.replace("\\", "\\\\")

    for node in nodes:
        node_text = node["text"]
        node_id = f"node_{abs(hash(node_text)) % (10 ** 8)}"

        node_type = node["type"]
        if isinstance(node_type, list):
            node_type = node_type[0]

        shape = node_shapes.get(node_type, "oval")

        src = node.get("source", [])
        if isinstance(src, str):
            src = [src]
        if "cmd" in src and "report" in src:
            border_color = "blue"  
            node_source_type = "merged"
        elif "cmd" in src:
            border_color = "red"   
            node_source_type = "cmd_only"
        else:
            border_color = "black" 
            node_source_type = "report_only"

        graph.node(
            node_id,
            label=safe_label(node_text),
            shape=shape,
            style="solid",
            color=border_color,
            fontcolor=border_color
        )
        added_nodes[node_text.lower()] = {"id": node_id, "source_type": node_source_type}

    for i, relation in enumerate(relations):
        source_node, source_type, target_node, target_type, relation_type = relation
        if relation_type == "Other":
            continue

        source_info = added_nodes.get(source_node.lower())
        target_info = added_nodes.get(target_node.lower())

        if source_info and target_info:
            src_type_1 = source_info["source_type"]
            src_type_2 = target_info["source_type"]

            if src_type_1 == "cmd_only" and src_type_2 == "cmd_only":
                edge_color = "red"  
            elif src_type_1 == "merged" and src_type_2 == "merged":
                edge_color = "blue"  
            elif (src_type_1 == "cmd_only" and src_type_2 == "merged") or (src_type_1 == "merged" and src_type_2 == "cmd_only"):
           
                edge_color = "red" if src_type_2 == "cmd_only" else "blue"
            else:
                edge_color = "black"  

            graph.edge(
                source_info["id"],
                target_info["id"],
                label=f"{i + 1}. {relation_type}",
                fontsize='10',
                color=edge_color,
                fontcolor=edge_color
            )

    output_path = os.path.join(os.getcwd(), output_filename)
    graph.render(output_path + ".dot", view=False)
    graph.save(output_path + ".dot")

def compare_entities_order(ent1, ent2, positions):
    pos_list1 = positions.get(ent1.lower(), [])
    pos_list2 = positions.get(ent2.lower(), [])

    if not pos_list1 and not pos_list2:
        return 0
    if not pos_list1:
        return 1
    if not pos_list2:
        return -1

    if min(pos_list1) < min(pos_list2):
        return -1
    elif min(pos_list1) > min(pos_list2):
        return 1
    else:
        return 0
def normalize_text_for_match(text):

    if not text:
        return ""
    text = text.lower().strip()
    text = text.replace("\\\\", "\\").replace("\\", "/")  
    text = re.sub(r'\s+', '', text)  

    text = re.sub(r'[^a-z0-9:/._%-]', '', text)
    return text
def build_entity_positions(text, entities):

    positions = {}
    text_norm = normalize_text_for_match(text) 

    for ent in entities:
        ent_norm = normalize_text_for_match(ent)
        if not ent_norm:
            positions[ent] = None
            continue

        match = re.search(re.escape(ent_norm), text_norm)
        if match:
            positions[ent] = match.start()
        else:
            positions[ent] = None

    return positions
def sort_relations_by_text_order(relations, positions):

    def relation_key(rel):
        e1, _, e2, _, _ = rel
        pos1 = positions.get(e1, None)
        pos2 = positions.get(e2, None)

        pos1 = pos1 if pos1 is not None else float('inf')
        pos2 = pos2 if pos2 is not None else float('inf')

        return (pos1, pos2)

    return sorted(relations, key=relation_key)

def process_single_CTI(model,report_path, grucialg_path,idx,regex_file,special_file, threshold=0.9, top_k=1):


    merged_nodes, merged_relations = build_merged_attack_graph(model,report_path, grucialg_path,idx,regex_file,special_file, threshold=0.9, top_k=1)
    
    with open(report_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    all_entities = [node["text"] for node in merged_nodes]

    entity_positions = build_entity_positions(full_text, all_entities)
 
    sorted_relations = sort_relations_by_text_order(merged_relations, entity_positions)

    return merged_nodes, sorted_relations

def batch_process_reports(models,report_dir, grucialg_path,regex_file, special_file,threshold=0.9, top_k=1):

    report_dir = Path(report_dir)
    if not report_dir.exists():
        return {"error""}

    grucialg_data = []
    with open(grucialg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  
                continue
            grucialg_data.append(json.loads(line))


    all_results = []
    for idx,detail in enumerate(grucialg_data):
        doc_key=detail["doc_key"]
        report_path = report_dir / f"{doc_key}.txt"
        print(report_path)
        nodes,relations=process_single_CTI(models,report_path, grucialg_path,idx,regex_file,special_file, threshold, top_k)
        
        draw_graph(nodes, relations, f"ClarityG/experiment/ag/graph_withTTP_cmd/{doc_key}")
        def drop_vec(nodes):
            for n in nodes:
                n.pop("vec", None)
            return nodes
        nodes = drop_vec(nodes)
        all_results.append({
            "doc_key": doc_key,
            "ners": nodes,
            "relations": relations
        })
    output_json = "ClarityG/experiment/ag/result_cmd_withTTP.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
 

    return all_results
def load_models():
    save_path_text = "ClarityG/TTP/result/text_roberta_0.5"
    text_model, tokenizer_text, id2label_text = init_text_model(save_path_text, model_path=os.path.join(save_path_text, "best_model_state.pt"))

    save_path_cmd = "ClarityG/TTP/result/dual_0.4"
    cmd_model, tokenizer_text_cmd, tokenizer_cmd, id2label_cmd = init_dual_model(
        save_path_cmd,
        model_path=os.path.join(save_path_cmd, "best_model_state.pt")
    )

    encoder_model, type_embeddings = load_encoder()

    return {
        "text_model": text_model,
        "tokenizer_text": tokenizer_text,
        "id2label_text": id2label_text,
        "cmd_model": cmd_model,
        "tokenizer_text_cmd": tokenizer_text_cmd,
        "tokenizer_cmd": tokenizer_cmd,
        "id2label_cmd": id2label_cmd,
        "encoder_model": encoder_model,
        "type_embeddings": type_embeddings
    }

if __name__ == "__main__":
    models = load_models()
    regex_file = "ClarityG/regexPattern.json"
    special_file = "ClarityG/datasets/NER_teshu.txt"
    report_dir="ClarityG/datasets/reports"
    grucialg_path= "ClarityG/experiment/ag/crucialg/test_AG_result_all.json"

    start_time = time.time()
    batch_process_reports(models,report_dir, grucialg_path,regex_file, special_file,threshold=0.9, top_k=1)
    end_time = time.time()
