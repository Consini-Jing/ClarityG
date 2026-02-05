import os
import sys
sys.path.append('..')
import json
import graphviz
import argparse

node_shapes = {
    'F': 'rect',             
    'O': 'diamond',          
    'P': 'ellipse'           
}

def read(json_file):
    docs = [json.loads(line) for line in open(json_file)]
    return docs

def get_new_ner_re(ns, rs):
    final_ner = []
    final_re = []
    re_nodes = set()
    for e in rs:
        re_nodes.add(e[0])
        re_nodes.add(e[3])
    ner_map = {}
    index=0
    for i in range(len(ns)):
        if ns[i][0] in re_nodes:
            final_ner.append(ns[i])
            ner_map[ns[i][0]]=index
            index+=1
    for re in rs:
        final_re.append([ner_map[re[0]],ner_map[re[3]],re[8]])
    return final_ner, final_re

def deduplicate_nodes(nodes,res):
    new_ner = []
    new_re = []
    node_pos ={}
    node_map_new={}
    index = 0
    for n in nodes:
        n_lower=n[4].lower()
        if n_lower not in node_pos:
            node_pos[n_lower]=n
            node_map_new[n_lower]=index
            index+=1
            new_ner.append(n)
            # print(n)
    for r in res:
        node_key1 = r[6].lower()
        node_key2 = r[7].lower()
        i_new1 = node_map_new[node_key1]
        i_new2 = node_map_new[node_key2]
        new_re.append([new_ner[i_new1][0],new_ner[i_new1][1],new_ner[i_new1][2],
                       new_ner[i_new2][0],new_ner[i_new2][1],new_ner[i_new2][2],
                      node_pos[node_key1][4],node_pos[node_key2][4],r[8],r[9]])
    return new_ner,new_re

def deduplicate_edges(res):
    new_re = []
    edge_pos = set()  
    index = 0
    for r in res:
        edge_tuple=(r[6],r[7],r[8])
        if edge_tuple not in edge_pos:
            edge_pos.add(edge_tuple)
            index +=1
            new_re.append([r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],index])
            index+=1
    return new_re

def draw_graph(new_nodes, new_edges, li):
    # print(li)
    graph_name_dot = li + ".dot"
    graph_name_dot_path = os.path.join(args.asg_reconstruction_graph, graph_name_dot)
    new_graph = graphviz.Digraph(graph_name_dot_path, filename=graph_name_dot)
    new_graph.body.extend(
        ['rankdir="LR"', 'size="9"', 'fixedsize="false"', 'splines="true"', 'nodesep=0.3', 'ranksep=0',
         'fontsize=10',
         'overlap="scalexy"',
         'engine= "neato"'])
    n_id = 0
    for n in new_nodes:
        if len(n) > 3:
            n[4] = n[4].replace('\\', '\\\\')
            new_graph.node(str(n_id), label=n[4], shape=node_shapes[n[2][1]])
        else:
            new_graph.node(str(n_id), label=n[2], shape=node_shapes[n[2][1]])
        n_id += 1
    e_id = 0
    for edge in new_edges:
        new_graph.edge(str(edge[0]), str(edge[1]), label=str(e_id) + '. ' + edge[2])
        e_id += 1

    new_graph.render(graph_name_dot_path, view=False)
    new_graph.save(graph_name_dot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_generator_json', type=str, default=None, required=True,
                        help="result from ner model")

    parser.add_argument('--asg_reconstruction_json', type=str, default=None, required=True,
                        help="ner screening results")
   
    parser.add_argument('--asg_reconstruction_graph', type=str, default=None, required=True,
                        help="ner screening results")
    args = parser.parse_args()
    re_predict_data = read(args.graph_generator_json)
    f = open(args.asg_reconstruction_json, 'w', encoding='utf-8')
    for d_js in re_predict_data:
        new_json = {}
        print(d_js['doc_key'])
        ners = d_js['ners']
        res = d_js['relations']
        indice = -1
        for i in range(len(ners)):
            if len(ners[i]) < 2:
                ners[i] = [indice, indice, ners[i][0]]
                indice -= 1
        new_ners,new_res=deduplicate_nodes(ners, res)
        new_res = deduplicate_edges(new_res)
        new_ners, new_res = get_new_ner_re(new_ners, new_res)
        draw_graph(new_ners, new_res, d_js['doc_key'])
        new_json['doc_key'] = d_js['doc_key']
        new_json['ners'] = new_ners
        new_json['relations'] = new_res
        json.dump(new_json, f)
        f.write('\n')
