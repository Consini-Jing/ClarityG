import shlex
import os
import json
import re
import graphviz
import networkx as nx
from collections import defaultdict
from test import predict_single

def read_commands_from_file(file_path):
    """
    从文本文件中读取命令行内容
    参数: file_path (str): 文本文件的路径
    返回: list: 包含所有非空命令行的列表
    """
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

def remove_all_quotes(cmd):
    quotes_to_remove = ['"', "'", '“', '”', '‘', '’']
    for q in quotes_to_remove:
        cmd = cmd.replace(q, '')
    return cmd
def cmd_split(cmd):
    cmd=remove_all_quotes(cmd)
    return shlex.split(cmd, posix=False)

# 实体识别
def load_patterns(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def match_entity(token, patterns):
    token = token.strip("\"'")
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
def extract_entities(tokens, patterns):
    entities = []
    for token in tokens:
        category = match_entity(token,patterns)
        if category is None:
            category = "other"
        entities.append([token, category])
    return entities
def generate_tagged_sentences(entities, valid_types={'process', 'file', 'socket'}):
    cmd = []
    pairs=[]
    sentence=""
    for i in range(len(entities)):
        e1_text, e1_type = entities[i]
        if e1_type !="process":
            continue
        for j in range(len(entities)):
            e2_text, e2_type = entities[j]
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

def process_command_line_single(command, regex_file="regexPattern.json"):
    """
    处理单句命令行,得到标注的命令行以及对应的标签,通过合理性以进行判断修复
    最后得到所有的关系，包括两个实体节点和对应的边
    """
    raw_relations=[]
    patterns = load_patterns(regex_file)
    cmd_list = cmd_split(command)
    entities_list = extract_entities(cmd_list, patterns)
    tagged_cmd = generate_tagged_sentences(entities_list)
    model_path="./R-BERT-master/model"
    for cmd in tagged_cmd:
        prediction,probs = predict_single(cmd, model_path)
        entity1_text,entity1_type,entity2_text,entity2_type,prediction_new =\
            verify_and_correct_relation(command,cmd,entities_list,prediction,probs)
        new_relation = [entity1_text, entity1_type, entity2_text, entity2_type, prediction_new]
        if new_relation not in raw_relations and new_relation[4] !="Other":
            raw_relations.append(new_relation)
    final_relations = detect_and_repairc_conflict_relations(command,raw_relations)
    final_relations = repair_other_relations_with_defaults(final_relations,entities_list)

    return final_relations

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
        return prediction  # 类型不支持，无修复

    # 获取实体在命令中的位置（基于tagged_cmd的e1/e2）
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
            # 清理特殊标记符号
            word_clean = word.lower().strip("<>/\\\"'=,;()[]")
            if word_clean in keywords:
                score += keywords[word_clean]
        relation_scores[rel] = score
        prob_dict = dict(zip(label_list, probs))
        fused_scores = {}
        for rel in relation_scores:
            # prob = probs.get(rel, 0)  # 原模型 softmax 概率
            prob = prob_dict.get(rel, 0)
            keyword_score = relation_scores[rel]
            fused_scores[rel] = 0.3 * keyword_score + 0.7 * prob  # 可调整权重比例

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
            rel_tuple = tuple(rel)
            if rel_tuple not in seen:
                seen.add(rel_tuple)
                unique_relations.append(rel)

        return unique_relations
    else:
        # 不是所有都是 Other，保持原样返回
        return relations

node_shapes = {
    'file': 'rect',             # 正方形
    'socket': 'diamond',          # 菱形
    'process': 'ellipse'           # 椭圆形（与圆形相同）
}
def draw_graph(relations,output_filename):
    """
       将多对关系绘制成图形，但过滤掉Other类型的关系
       参数:
           relations: 关系列表，每个元素格式为 [源节点, 源类型, 目标节点, 目标类型, 关系类型]
           output_filename: 输出文件名(不带扩展名)
       """
    graph = graphviz.Digraph(output_filename, filename=output_filename + ".dot")

    graph.attr(
        rankdir='LR',  # 从左到右布局
        size='9',  # 图形大小
        splines='true',  # 使用曲线边
        nodesep='0.3',  # 节点间距
        ranksep='0.5',  # 层级间距
        fontsize='10',  # 字体大小
        overlap='false',  # 防止节点重叠
        engine='dot'  # 更适合复杂图的布局引擎
    )

    added_nodes = {}

    for i, relation in enumerate(relations):
        source_node, source_type, target_node, target_type, relation_type = relation

        if relation_type=="Other":
            continue  # 跳过Other类型的关系

        source_label = source_node.replace('\\', '\\\\')
        target_label = target_node.replace('\\', '\\\\')

        source_id = f"node_{hash(source_node)}"
        target_id = f"node_{hash(target_node)}"

        if source_id not in added_nodes:
            graph.node(
                source_id,
                label=source_label,
                shape=node_shapes[source_type]
            )
            added_nodes[source_id] = True

        if target_id not in added_nodes:
            graph.node(
                target_id,
                label=target_label,
                shape=node_shapes[target_type]
            )
            added_nodes[target_id] = True

        graph.edge(
            source_id,
            target_id,
            label=f"{i + 1}. {relation_type}",
            fontsize='9'
        )

    output_path = os.path.join(os.getcwd(), output_filename)
    graph.render(output_path + ".dot", view=False)
    graph.save(output_path + ".dot")

if __name__ == "__main__":
    commands=r"C:\Windows\system32\cmd.exe /c cd %appdata%\Microsoft\Network && powershell Expand-Archive python-3.10.4-embed-amd64.zip -DestinationPath %appdata%\Microsoft\Network"
    results=process_command_line_single(commands)
    print(results)
    # draw_graph(results,"aaa")
