import json
import os
from pathlib import Path

def convert_crucialg_ag(
    input_path: str,
    output_path: str
):
    """
    将 CrucialG 输出的 JSONL 攻击图文件
    转换为统一的攻击图结构（list[dict]）
    """

    # ===== 实体类型映射（按你给的）=====
    type_mapping = {
        "SO": "socket",
        "TP": "process",
        "MP": "process",
        "SF": "file",
        "TF": "file",
        "MF": "file",
    }

    # ===== 关系类型映射（可按需扩展）=====
    relation_mapping = {
        "RD": "read",  # Read
        "WR": "write",  # Write
        "EX": "exec",  # Execute
        "CD": "chmod",  # Change permissions
        "UK": "unlink",  # Delete / Unknown (兼容删除)
        "ST": "send",  # Socket Transmit
        "RF": "receive",  # Receive from socket
        "FR": "fork",  # Fork
        "IJ": "inject",
    }

    def map_entity_type(label: str) -> str:
        """实体类型映射（带兜底）"""
        return type_mapping.get(label, f"unknown_{label.lower()}")

    def map_relation_type(label: str) -> str:
        """关系类型映射（带兜底）"""
        return relation_mapping.get(label, label.lower())

    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            doc_key = data.get("doc_key")
            raw_ners = data.get("ners", [])
            raw_relations = data.get("relations", [])

            # ===== 1. 实体转换 =====
            ners = []
            for ner in raw_ners:
                # 原格式: [start, end, label, score, text]
                if len(ner) < 5:
                    continue

                label = ner[2]
                text = ner[4]

                ners.append({
                    "text": text,
                    "type": map_entity_type(label)
                })

            # ===== 2. 关系转换 =====
            relations = []
            for rel in raw_relations:
                # 原格式: [head_idx, tail_idx, rel_type]
                if len(rel) < 3:
                    continue

                head_idx, tail_idx, rel_type = rel

                if head_idx >= len(raw_ners) or tail_idx >= len(raw_ners):
                    continue

                head_ner = raw_ners[head_idx]
                tail_ner = raw_ners[tail_idx]

                head_text = head_ner[4]
                head_type = map_entity_type(head_ner[2])

                tail_text = tail_ner[4]
                tail_type = map_entity_type(tail_ner[2])

                relations.append([
                    head_text,
                    head_type,
                    tail_text,
                    tail_type,
                    map_relation_type(rel_type)
                ])

            results.append({
                "doc_key": doc_key,
                "ners": ners,
                "relations": relations
            })

    # ===== 输出为标准 JSON（非 JSONL）=====
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✔ 转换完成，共处理 {len(results)} 篇报告")



def evaluate(process_path: str, predicate_path: str, result_path: str):
    """
    对比我们方法 vs CRUcialG 攻击图
    输出 TSV 格式，每篇报告一行，最后一行输出 Average
    """

    def load_ag(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    our_ags = load_ag(predicate_path)
    crucialg_ags = load_ag(process_path)

    our_map = {item["doc_key"]: item for item in our_ags}
    crucialg_map = {item["doc_key"]: item for item in crucialg_ags}

    total_docs = 0
    sum_our_nodes = sum_crucialg_nodes = 0
    sum_our_edges = sum_crucialg_edges = 0

    rows = []

    for doc_key in our_map:
        if doc_key not in crucialg_map:
            continue

        our_ag = our_map[doc_key]
        crucialg_ag = crucialg_map[doc_key]

        our_nodes = len(our_ag.get("ners", []))
        crucialg_nodes = len(crucialg_ag.get("ners", []))
        node_diff = our_nodes - crucialg_nodes

        our_edges = len(our_ag.get("relations", []))
        crucialg_edges = len(crucialg_ag.get("relations", []))
        edge_diff = our_edges - crucialg_edges

        # 累计
        sum_our_nodes += our_nodes
        sum_crucialg_nodes += crucialg_nodes
        sum_our_edges += our_edges
        sum_crucialg_edges += crucialg_edges
        total_docs += 1

        # 每篇报告一行
        rows.append([
            doc_key,
            crucialg_nodes,
            our_nodes,
            node_diff,
            crucialg_edges,
            our_edges,
            edge_diff
        ])

    # Average 行
    if total_docs > 0:
        rows.append([
            "Average",
            round(sum_crucialg_nodes / total_docs, 2),
            round(sum_our_nodes / total_docs, 2),
            round((sum_our_nodes - sum_crucialg_nodes) / total_docs, 2),
            round(sum_crucialg_edges / total_docs, 2),
            round(sum_our_edges / total_docs, 2),
            round((sum_our_edges - sum_crucialg_edges) / total_docs, 2),
        ])

    # 写入 TSV 文件
    with open(result_path, "w", encoding="utf-8") as f:
        # 写表头
        f.write("CTI_Report\tCRUcialG_Nodes\tClarityG_Nodes\tNode_Diff\tCRUcialG_Edges\tClarityG_Edges\tEdge_Diff\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

    print(f"✔ TSV 输出完成，每篇报告一行，结果已保存到: {result_path}")



if __name__ == "__main__":
    predicate_path="/root/ClarityG/experiment/ag/result_cmd_withTTP.json"
    process_file="/root/ClarityG/experiment/ag/crucialg/AG_crucialg.json"
    result_path="/root/ClarityG/experiment/ag/crucialg/result_evaluate.tsv"
    evaluate(process_file,predicate_path,result_path)


