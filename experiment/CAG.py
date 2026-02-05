from experiment.NER_RE import process_command_line_single
from test import predict_single
from itertools import zip_longest
import os
import json
import re
import networkx as nx
from collections import defaultdict

def build_rule_command_graph(cmd_gt, regex_file,special_file, output_file=None):

    graphs = []
    with open(cmd_gt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cid = data.get("cid", "")
            command = data.get("cmd", "")

            cmd_text, entities_list, final_relations = command_analysis_rule(command, regex_file, special_file)

            nodes = {}
            for ent_text, ent_type in entities_list:
                if ent_type=="other":
                    continue

                key = (ent_text, ent_type)
                if key not in nodes:
                    nodes[key] = {
                        "id": f"n{len(nodes) + 1}",
                        "text": ent_text,
                        "type": ent_type
                    }
            edges = []
            for rel in final_relations:
                e1_text, e1_type, e2_text, e2_type, rel_type = rel
                source_id = nodes.get((e1_text, e1_type), {}).get("id")
                target_id = nodes.get((e2_text, e2_type), {}).get("id")
                if source_id and target_id:
                    edges.append({"source": source_id, "target": target_id, "type": rel_type})

            graph = {
                "cid": cid,
                "cmd": command,
                "nodes": list(nodes.values()),
                "edges": edges
            }

            graphs.append(graph)
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for g in graphs:
                f_out.write(json.dumps(g, ensure_ascii=False) + "\n")
    return graphs

def command_analysis_rule(command, regex_file,special_file):
    entities,tagged_cmd = process_command_line_single(command, regex_file,special_file)


    entities_list = [[t, tp] for t, tp in entities]
    model_path = "ClarityG/R-BERT-master/model"
    raw_relations = []
    for cmd in tagged_cmd:
        prediction,probs = predict_single(cmd, model_path)

        e1_match = re.search(r"<e1>(.+?)</e1>", cmd)
        e1_text = e1_match.group(1) if e1_match else ""
        def find_ent_type(text, entities_list):

            for ent, tp in entities_list:
                if ent.strip().lower() == text.strip().lower():
                    return tp
            return None
        e1_type = find_ent_type(e1_text, entities_list)
        if(prediction!="Other" and e1_type=="process"):

            entity1_text,entity1_type,entity2_text,entity2_type,prediction_new =verify_and_correct_relation(command,cmd,entities_list,prediction,probs)
            new_relation = [entity1_text, entity1_type, entity2_text, entity2_type, prediction_new]

            if new_relation not in raw_relations and new_relation[4] !="Other":
                raw_relations.append(new_relation)

    final_relations = detect_and_repairc_conflict_relations(command,raw_relations)
    final_relations = repair_other_relations_with_defaults(final_relations,entities_list)

    for i, rel in enumerate(final_relations):
        rel_type = rel[4]
        if rel_type != "Other" and "(e1,e2)" not in rel_type:
            rel_type += "(e1,e2)"
            final_relations[i][4] = rel_type

    return command,entities_list,final_relations

def build_command_graph(cmd_gt, regex_file,special_file, output_file=None):

    graphs = []
    with open(cmd_gt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cid = data.get("cid", "")
            command = data.get("cmd", "")

            cmd_text, entities_list, final_relations = command_analysis(command, regex_file, special_file)

            nodes = {}
            for ent_text, ent_type in entities_list:
                if ent_type=="other":
                    continue

                key = (ent_text, ent_type)
                if key not in nodes:
                    nodes[key] = {
                        "id": f"n{len(nodes) + 1}",
                        "text": ent_text,
                        "type": ent_type
                    }
            edges = []
            for rel in final_relations:
                e1_text, e1_type, e2_text, e2_type, rel_type = rel
                source_id = nodes.get((e1_text, e1_type), {}).get("id")
                target_id = nodes.get((e2_text, e2_type), {}).get("id")
                if source_id and target_id:
                    edges.append({"source": source_id, "target": target_id, "type": rel_type})

            graph = {
                "cid": cid,
                "cmd": command,
                "nodes": list(nodes.values()),
                "edges": edges
            }

            graphs.append(graph)
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for g in graphs:
                f_out.write(json.dumps(g, ensure_ascii=False) + "\n")
    return graphs

def command_analysis(command, regex_file,special_file):

    entities,tagged_cmd = process_command_line_single(command, regex_file,special_file)

    entities_list = [[t, tp] for t, tp in entities]
    model_path = "ClarityG/R-BERT-master/model"
    raw_relations = []
    for cmd in tagged_cmd:
        prediction,probs = predict_single(cmd, model_path)
        if(prediction=="Other"):
            continue
     
        e1_match = re.search(r"<e1>(.+?)</e1>", cmd)
        e1_text = e1_match.group(1) if e1_match else ""
        e2_match = re.search(r"<e2>(.+?)</e2>", cmd)
        e2_text = e2_match.group(1) if e2_match else ""
  
        def find_ent_type(text, entities_list):
            for ent, tp in entities_list:
                if ent.strip().lower() == text.strip().lower():
                    return tp
            return "other"       
        e1_type = find_ent_type(e1_text, entities_list)
        e2_type = find_ent_type(e2_text, entities_list)

  
        rel = [e1_text, e1_type, e2_text, e2_type, prediction]
        if rel not in raw_relations:
            raw_relations.append(rel)
    print(entities_list,raw_relations)
    return command,entities_list,raw_relations

def verify_and_correct_relation(command,tagged_cmd,entities_list,prediction,probs):

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
    e1_match = re.search(r"<e1>(.+?)</e1>", tagged_cmd)
    e2_match = re.search(r"<e2>(.+?)</e2>", tagged_cmd)
    e1_text = e1_match.group(1) if e1_match else ""
    e2_text = e2_match.group(1) if e2_match else ""

    def find_entity_type(entity_text, entities_list):
        for ent, ent_type in entities_list:
            if ent.strip().lower() == entity_text.strip().lower():
                return ent_type
        return None
    e1_type = find_entity_type(e1_text, entities_list)
    e2_type = find_entity_type(e2_text, entities_list)
    expected_types = relation_entity_types.get(prediction)
    if prediction=="Other" or expected_types!=(e1_type,e2_type):
        e1_text,e1_type,e2_text,e2_type,prediction=rule_based_relation_fix(command,tagged_cmd,e1_text,e1_type,e2_text,e2_type,prediction,probs)

    return e1_text,e1_type,e2_text,e2_type,prediction
def rule_based_relation_fix(command,tagged_cmd,e1_text,e1_type,e2_text,e2_type,prediction,probs):
    window_size=3
    candidates = []
    type_pair = (e1_type, e2_type)
    valid_relations = {
        ('process', 'file'): [
            'process-file-read', 'process-file-write', 'process-file-exec',
            'process-file-chmod', 'process-file-unlink'
        ],
        ('process', 'socket'): [
            'process-socket-send', 'process-socket-receive'
        ],
        ('process', 'process'): [
            'process-process-fork', 'process-process-inject', 'process-process-unlink'
        ]
    }
    relation_keyword_weight = {
        'process-file-read': {
            'cat': 1.0, 'type': 1.0, 'less': 0.8, 'more': 0.8,
            'read': 1.0, 'open': 0.6, 'head': 0.6, 'tail': 0.6,
            'get-content': 0.9, 'findstr': 0.5, 'grep': 0.5
        },
        'process-file-write': {
            'write': 1.0, 'echo': 0.9, '>': 0.9, '>>': 0.9,
            'tee': 0.7, 'set-content': 1.0, 'copy': 0.6, 'out-file': 0.8,
            'move': 0.5, 'upload': 0.5
        },
        'process-file-exec': {
            'run': 1.0, 'start': 1.0, 'exec': 1.0, 'launch': 0.9,
            '.exe': 0.9, 'powershell': 0.6, 'cmd': 0.6,
            'sh': 0.5, 'execute': 1.0, '/c': 0.5
        },
        'process-file-chmod': {
            'chmod': 1.0, 'attrib': 0.9, 'icacls': 0.8, 'setfacl': 0.8,
            'takeown': 0.7, 'cacls': 0.8
        },
        'process-file-unlink': {
            'rm': 1.0, 'del': 1.0, 'erase': 0.8, 'unlink': 1.0,
            'remove-item': 1.0, 'delete': 0.9, 'rmdir': 0.7
        },
        'process-socket-send': {
            'send': 1.0, 'post': 0.9, 'curl': 1.0, 'wget': 1.0,
            'ftp': 1.0, 'scp': 0.8, 'netcat': 0.8, 'nc': 0.9,
            'socket.send': 0.9, 'powershell -c Invoke-WebRequest': 0.9
        },
        'process-socket-receive': {
            'recv': 1.0, 'listen': 1.0, 'netcat': 0.9, 'nc': 0.9,
            'socket.receive': 1.0, 'accept': 0.8, 'bind': 0.6
        },
        'process-process-fork': {
            'fork': 1.0, 'spawn': 1.0, 'clone': 1.0,
            'createprocess': 1.0, 'cmd /c': 0.7, 'new-process': 0.9,
            'powershell start-process': 0.9
        },
        'process-process-inject': {
            'inject': 1.0, 'rundll32': 1.0, 'createremotethread': 1.0,
            'reflective': 0.9, 'hollow': 0.9, 'dllinject': 1.0,
            'remote thread': 1.0, 'shellcode': 0.9
        },
        'process-process-unlink': {
            'kill': 1.0, 'terminate': 1.0, 'taskkill': 1.0,
            'pkill': 0.9, 'stop-process': 1.0, 'end': 0.7
        }
    }
    label_list = [
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
    if type_pair not in valid_relations:
        return prediction  

    tokens = tagged_cmd.split()
    e1_idx = next((i for i, tok in enumerate(tokens) if "<e1>" in tok), -1)
    e2_idx = next((i for i, tok in enumerate(tokens) if "<e2>" in tok), -1)

    relation_scores = {}
    for rel in valid_relations[type_pair]:
        keywords = relation_keyword_weight.get(rel, {})
        score = 0.0
        e1_window = tokens[max(0, e1_idx - window_size): e1_idx + window_size + 1]
        e2_window = tokens[max(0, e2_idx - window_size): e2_idx + window_size + 1]
        context_window = set(e1_window + e2_window)
        for word in context_window:
            word_clean = word.lower().strip("<>/\\\"'=,;()[]")
            if word_clean in keywords:
                score += keywords[word_clean]
        relation_scores[rel] = score
        prob_dict = dict(zip(label_list, probs))
        fused_scores = {}
        for rel in relation_scores:
            prob = prob_dict.get(rel, 0)
            keyword_score = relation_scores[rel]
            fused_scores[rel] = 0.3 * keyword_score + 0.7 * prob

    best_relation, best_score = max(fused_scores.items(), key=lambda x: x[1])
    threshold = 0.5
    if best_score < threshold:
        prediction_new = 'Other'
    else:
        prediction_new = best_relation
    return e1_text,e1_type,e2_text,e2_type,prediction_new

def detect_and_repairc_conflict_relations(command,relations):
    G = nx.DiGraph()
    edge_to_relations = defaultdict(list)
    for rel in relations:
        e1_text, e1_type, e2_text, e2_type, label = rel
        G.add_edge(e1_text, e2_text)
        edge_to_relations[(e1_text, e2_text)].append(rel)

    to_remove = set()
    for (a, c) in G.edges():
        if a == c:
            continue
        try:
            for path in nx.all_simple_paths(G, source=a, target=c, cutoff=3):
                if len(path) >= 3:
                    to_remove.add((a, c))
                    break
        except nx.NetworkXNoPath:
            continue


    cleaned_relations = []
    for (e1, e2), rel_list in edge_to_relations.items():
        if (e1, e2) not in to_remove:
            cleaned_relations.extend(rel_list)

    return cleaned_relations

def repair_other_relations_with_defaults(relations, entities_list):
    if all(rel[4] == "Other" for rel in relations):
        new_relations = []
        entities_sorted = entities_list
        process_positions = [(i, e) for i, e in enumerate(entities_sorted) if e[1] == 'process']

        relation_map = {
            ('process', 'file'): 'process-file-exec',
            ('process', 'socket'): 'process-socket-send',
            ('process', 'process'): 'process-process-fork'
        }

        for idx, ent in enumerate(entities_sorted):
            if ent[1] == 'process':
               
                prev_procs = [p for p in process_positions if p[0] < idx]
                if prev_procs:
                    nearest_proc = prev_procs[-1][1]
                    new_relations.append([
                        nearest_proc[0], nearest_proc[1],
                        ent[0], ent[1],
                        relation_map[('process', 'process')]
                    ])
            elif ent[1] in ['file', 'socket']:
                
                prev_procs = [p for p in process_positions if p[0] < idx]
                if prev_procs:
                    nearest_proc = prev_procs[-1][1]
                    rel_label = relation_map.get((nearest_proc[1], ent[1]), 'Other')
                    new_relations.append([
                        nearest_proc[0], nearest_proc[1],
                        ent[0], ent[1],
                        rel_label
                    ])

        unique_relations = []
        seen = set()
        for rel in new_relations:
            rel_tuple = tuple(rel)  #
            if rel_tuple not in seen:
                seen.add(rel_tuple)
                unique_relations.append(rel)

        return unique_relations
    else:
        return relations

def evaluate_attack_graphs(pred_file, gt_file):

    pred_graphs = [json.loads(line) for line in open(pred_file, 'r', encoding='utf-8') if line.strip()]
    gt_graphs = [json.loads(line) for line in open(gt_file, 'r', encoding='utf-8') if line.strip()]

    node_tp_total = node_fp_total = node_fn_total = 0
    edge_tp_total = edge_fp_total = edge_fn_total = 0
    graph_prec_total = graph_recall_total = 0
    graphsim_total = 0
    count = 0

    for pred, gt in zip_longest(pred_graphs, gt_graphs):
        if pred is None or gt is None:
            continue

    
        pred_nodes = set((n['text'], n['type']) for n in pred['nodes'])
        gt_nodes = set((n['text'], n['type']) for n in gt['nodes'])
        node_tp = len(pred_nodes & gt_nodes)
        node_fp = len(pred_nodes - gt_nodes)
        node_fn = len(gt_nodes - pred_nodes)
        node_tp_total += node_tp
        node_fp_total += node_fp
        node_fn_total += node_fn


        pred_id_map = {n['id']: (n['text'], n['type']) for n in pred['nodes']}
        gt_id_map = {n['id']: (n['text'], n['type']) for n in gt['nodes']}
        pred_edges_set = set((pred_id_map[e['source']], pred_id_map[e['target']], e['type']) for e in pred['edges'])
        gt_edges_set = set((gt_id_map[e['source']], gt_id_map[e['target']], e['type']) for e in gt['edges'])
        edge_tp = len(pred_edges_set & gt_edges_set)
        edge_fp = len(pred_edges_set - gt_edges_set)
        edge_fn = len(gt_edges_set - pred_edges_set)
        edge_tp_total += edge_tp
        edge_fp_total += edge_fp
        edge_fn_total += edge_fn

        graph_prec = (node_tp + edge_tp) / max(len(pred_nodes)+len(pred_edges_set), 1)
        graph_recall = (node_tp + edge_tp) / max(len(gt_nodes)+len(gt_edges_set), 1)
        graph_prec_total += graph_prec
        graph_recall_total += graph_recall

        pred_graph_set = pred_nodes | pred_edges_set  
        gt_graph_set = gt_nodes | gt_edges_set

        inter = len(pred_graph_set & gt_graph_set)
        union = len(pred_graph_set | gt_graph_set)

        if union>0:
            graphsim = inter / union
        else:
            graphsim = 1.0
        graphsim_total += graphsim

        count += 1

    def calc_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp+fp) else 0
        recall = tp / (tp + fn) if (tp+fn) else 0
        f1 = 2*precision*recall/(precision+recall+1e-8)
        accuracy = tp / (tp + fp + fn) if (tp+fp+fn) else 0
        return precision, recall, accuracy, f1

    node_metrics = calc_metrics(node_tp_total, node_fp_total, node_fn_total)
    edge_metrics = calc_metrics(edge_tp_total, edge_fp_total, edge_fn_total)
    graph_precision_avg = graph_prec_total / count
    graph_recall_avg = graph_recall_total / count
    graph_f1_avg = 2 * graph_precision_avg * graph_recall_avg / (graph_precision_avg + graph_recall_avg + 1e-8)
    graph_accuracy_avg = (graph_precision_avg + graph_recall_avg)/2  
    graphsim_avg = graphsim_total / count

    metrics = {
        "Node": {"Precision": node_metrics[0], "Recall": node_metrics[1], "Accuracy": node_metrics[2], "F1": node_metrics[3]},
        "Edge": {"Precision": edge_metrics[0], "Recall": edge_metrics[1], "Accuracy": edge_metrics[2], "F1": edge_metrics[3]},
        "Graph": {"Precision": graph_precision_avg, "Recall": graph_recall_avg, "Accuracy": graph_accuracy_avg, "F1": graph_f1_avg,"GraphSim": graphsim_avg}
    }

    for k, v in metrics.items():
        print(f"{k} Metrics:")
        for key, val in v.items():
            print(f"  {key}: {val:.4f}")
    return metrics


if __name__ == "__main__":

    ground_truth_file = "ClarityG/experiment/data/cmd_graph_GT.jsonl"
    regex_file = "ClarityG/regexPattern.json"
    special_file = "ClarityGa/datasets/NER_teshu.txt"
    output_tmp_file = "ClarityG/experiment/data/GT.jsonl"


    # build_rule_command_graph(ground_truth_file, regex_file, special_file, output2_file)
    # evaluate_attack_graphs(ground_truth_file,output2_file)
