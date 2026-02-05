import json


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

def normalize_for_match(s: str):
    return s.strip().lower().replace("\\", "/")

def process_block(block_lines, cid):
    raw_cmd = block_lines[0].strip()
    nodes = []
    edges = []
    id_map = {}
    node_id_counter = 1


    for line in block_lines[1:]:
        if not line.strip():
            continue
        if "\t" not in line:

            continue
        relation_type, rest = line.split("\t", 1)
        if relation_type == "Other":
            continue  

        if "<e1>" not in rest or "<e2>" not in rest:
            continue

        ent1_start = rest.index("<e1>") + 4
        ent1_end = rest.index("</e1>")
        ent2_start = rest.index("<e2>") + 4
        ent2_end = rest.index("</e2>")

        ent1 = rest[ent1_start:ent1_end].strip()
        ent2 = rest[ent2_start:ent2_end].strip()

 
        type1, type2 = relation_entity_types.get(relation_type, ("unknown", "unknown"))


        key1 = (normalize_for_match(ent1), type1)
        key2 = (normalize_for_match(ent2), type2)

        if key1 in id_map:
            id1 = id_map[key1]
        else:
            id1 = f"n{node_id_counter}"
            nodes.append({"id": id1, "text": ent1, "type": type1})
            id_map[key1] = id1
            node_id_counter += 1

        if key2 in id_map:
            id2 = id_map[key2]
        else:
            id2 = f"n{node_id_counter}"
            nodes.append({"id": id2, "text": ent2, "type": type2})
            id_map[key2] = id2
            node_id_counter += 1

        edges.append({
            "source": id1,
            "target": id2,
            "type": relation_type
        })

    return {
        "cid": cid,
        "cmd": raw_cmd,
        "nodes": nodes,
        "edges": edges
    }

def convert_dataset(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    all_data = []

    for i, blk in enumerate(blocks):
        blk_lines = blk.strip().split("\n")
        if not blk_lines:
            continue
        data = process_block(blk_lines, cid=f"{i+1:04d}")
        all_data.append(data)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "ClarityG/experiment/data/cmd_relation.tsv"
    output_file = "ClarityG/experiment/data/cmd_graph_GT.jsonl"
    convert_dataset(input_file, output_file)
