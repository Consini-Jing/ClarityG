import json
import os
from pathlib import Path

def convert_crucialg_ag(
    input_path: str,
    output_path: str
):
 
    type_mapping = {
        "SO": "socket",
        "TP": "process",
        "MP": "process",
        "SF": "file",
        "TF": "file",
        "MF": "file",
    }

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
    }

    def map_entity_type(label: str) -> str:
     
        return type_mapping.get(label, f"unknown_{label.lower()}")

    def map_relation_type(label: str) -> str:
    
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

        
            ners = []
            for ner in raw_ners:

                if len(ner) < 5:
                    continue

                label = ner[2]
                text = ner[4]

                ners.append({
                    "text": text,
                    "type": map_entity_type(label)
                })

            relations = []
            for rel in raw_relations:
          
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

     with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)




def evaluate(process_path: str, predicate_path: str, result_path: str):


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

  
        sum_our_nodes += our_nodes
        sum_crucialg_nodes += crucialg_nodes
        sum_our_edges += our_edges
        sum_crucialg_edges += crucialg_edges
        total_docs += 1

      
        rows.append([
            doc_key,
            crucialg_nodes,
            our_nodes,
            node_diff,
            crucialg_edges,
            our_edges,
            edge_diff
        ])

    
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

  
    with open(result_path, "w", encoding="utf-8") as f:
    
        f.write("CTI_Report\tCRUcialG_Nodes\tClarityG_Nodes\tNode_Diff\tCRUcialG_Edges\tClarityG_Edges\tEdge_Diff\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

 


if __name__ == "__main__":
    predicate_path="ClarityG/experiment/ag/result_cmd_withTTP.json"
    process_file="ClarityG/experiment/ag/crucialg/AG_crucialg.json"
    result_path="ClarityG/experiment/ag/crucialg/result_evaluate.tsv"
    evaluate(process_file,predicate_path,result_path)


