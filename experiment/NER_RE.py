import shlex
import os
import json
import re
from collections import defaultdict
from test import predict_single
from NER_REGEX import load_ground_truth_detailed,evaluate_with_details

def strip_multiple_edge_quotes(s: str) -> str:

    if not s:
        return s
    s = s.strip()
    s = s.lstrip('"\'')  # 去掉开头的所有 ' 或 "
    s = s.rstrip('"\'')  # 去掉结尾的所有 ' 或 "
    return s
def read_commands_from_file(file_path):

    commands = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stripped_line = line.strip()     # 去除首尾空白字符
                if stripped_line: # 跳过空行
                    commands.append(stripped_line)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None
    return commands

def load_patterns(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# 生成标注样本
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
def load_special_phrases(special_file):
    """加载特殊短语文件"""
    special_phrases = {}
    try:
        with open(special_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    phrase, entity_type = line.split('\t', 1)
                    special_phrases[phrase] = entity_type
        # print(special_phrases)
        '''
        {
          'active directory web services': 'File', 
          'internet explorer': 'File',  ...
        }
        '''
        # print(f"成功加载 {len(special_phrases)} 个特殊短语")
        return special_phrases
    except Exception as e:
        print(f"加载特殊短语文件失败: {e}")
        return {}

def extract_entities_from_command(command, patterns, special_phrases):
    """
    从单个命令中提取实体
    输入: 命令行字符串, 正则模式文件路径, 特殊短语字典
    输出: [(实体内容, 实体类型), ...]
    可能存在重复的实体
    """
    entities=[]
    used_positions = set()  # 记录已匹配的位置

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
                # 检查这个位置是否已经被其他实体占用
                overlap = False
                for used_start, used_end in used_positions:
                    if not (pos >= used_end or pos + len(phrase) <= used_start):
                        overlap = True
                        break

                if not overlap:
                    # 标记这个位置已被使用
                    used_positions.add((pos, pos + len(phrase)))
                    # 记录位置
                    entities.append((pos, phrase, entity_type.lower()))

            start = pos + 1

    if used_positions:
        sorted_positions = sorted(used_positions)
        segments = []
        last_end = 0

        for start, end in sorted_positions:
            if last_end < start:
                segments.append((last_end, command[last_end:start]))
            last_end = end

        if last_end < len(command):
            segments.append((last_end, command[last_end:]))

        for seg_start, segment in segments:
            if segment.strip():
                offset = seg_start
                for token in segment.split():
                    token_pos = command.find(token, offset)
                    category = match_entity(strip_multiple_edge_quotes(token), patterns)
                    if category and category != "other":
                        entities.append((token_pos, token, category))
                    else:
                        entities.append((token_pos, token, "other"))
                    offset = token_pos + len(token)
    else:
        offset = 0
        for token in command.split():
            token_pos = command.find(token, offset)
            category = match_entity(strip_multiple_edge_quotes(token), patterns)
            if category and category != "other":
                entities.append((token_pos, token, category))
            else:
                entities.append((token_pos, token, "other"))
            offset = token_pos + len(token)

    entities.sort(key=lambda x: x[0])


    if entities:
        idx, text, typ = entities[0]
        entities[0] = (idx, text, "process")

    return [(text, typ) for _, text, typ in entities]

# 单条命令行提取关系
def process_command_line_single(command,regex_file,special_file):

    raw_relations=[]

    patterns = load_patterns(regex_file)

    special_phrases = load_special_phrases(special_file)

    entities = extract_entities_from_command(command, patterns, special_phrases)

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
            e1 = re.search(r"<e1>(.*?)</e1>", cmd).group(1)  #提取文本
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

def infer_entity_types_by_relation_single(cmd_file,regex_file,special_file):

    with open(regex_file, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
    try:
        with open(cmd_file, 'r', encoding='utf-8') as f:
            commands = [line.strip() for line in f if line.strip()]
        print(f"成功读取 {len(commands)} 个命令")
    except Exception as e:
        print(f"读取命令行文件失败: {e}")
        return []
    results = []

    for i, command in enumerate(commands):
        entities, _  = process_command_line_single(command,regex_file,special_file)
        results.append((command, entities))

    print(f"处理完成！总共处理了 {len(commands)} 个命令")

    return results

    pass
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

    ground_truth_file = "/root/ClarityG/datasets/NER_bio.txt"
    cmd_file = "/root/ClarityG/datasets/NER_cmd.txt"
    regex_file = "/root/ClarityG/regexPattern.json"
    special_file = "/root/ClarityG/datasets/NER_teshu.txt"

    ground_truth = load_ground_truth_detailed(ground_truth_file)
    regex_results = infer_entity_types_by_relation_single(cmd_file, regex_file,special_file)

    p_all, r_all, acc_all, f1_all, \
        p_target, r_target, acc_target, f1_target, details = evaluate_with_details(ground_truth, regex_results)

    print(f"总的指标 (包含 all / other): Precision={p_all:.3f}, Recall={r_all:.3f}, F1={f1_all:.3f}")
    print(f"总的指标 (仅目标实体): Precision={p_target:.3f}, Recall={r_target:.3f}, F1={f1_target:.3f}")

    metrics_by_type = compute_entity_metrics_by_type(details)

    print("\n每类实体类型评估指标:")
    print(f"{'Type':15} {'Precision':10} {'Recall':10} {'F1-score':10}")
    for typ, m in metrics_by_type.items():
        print(f"{typ:15} {m['precision']:.3f}     {m['recall']:.3f}     {m['f1']:.3f}")