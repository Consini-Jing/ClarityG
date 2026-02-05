import json
import os
import torch
from typing import Dict, List, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from collections import defaultdict
import argparse
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, ARGVA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64         
HIDDEN_DIM = EMBED_DIM * 2
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 4

def to_pyg_data(doc):

    ners = doc.get("ners", [])
    relations = doc.get("relations", [])


    node_index = {}
    x_features = []

    node_map = {"process":0, "file":1, "socket":2} 

    for idx, ner in enumerate(ners):
        node_index[ner["text"].lower()] = idx
        t = ner["type"].lower()
        x_features.append(node_map.get(t, 3))  

    x = torch.eye(len(node_map))[x_features] 

  
    edge_index = []
    edge_attr = []

    edge_map = {"read":0, "write":1, "fork":2,"send":3, "receive":4,"exec":5, "chmod":6,
                "unlike":7, "inject":8}  

    for rel in relations:
        if len(rel) < 5:
            continue
        h_text, h_type, t_text, t_type, r_type = rel
        if h_text.lower() not in node_index or t_text.lower() not in node_index:
            continue
        src = node_index[h_text.lower()]
        tgt = node_index[t_text.lower()]
        etype = edge_map.get(r_type.lower(), len(edge_map))  

        edge_index.append([src, tgt])
        edge_index.append([tgt, src])
        edge_attr.append([etype])
        edge_attr.append([etype])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#Graph Encoder（GCN）
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


def train_autoencoder(datalist, feature_dim):
    dataloader = DataLoader(datalist, batch_size=BATCH_SIZE, shuffle=True)

    encoder = Encoder(feature_dim, HIDDEN_DIM, EMBED_DIM)
    discriminator = Linear(EMBED_DIM, EMBED_DIM)

    model = ARGVA(encoder, discriminator).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0

        for data in dataloader:
            data = data.to(device)
            model.train()
            optimizer.zero_grad()

            z = model.encode(data.x, data.edge_index)

     
            loss = model.recon_loss(z, data.edge_index)
            loss += model.reg_loss(z)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

    return model


def get_graph_embedding(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.edge_index.to(device))
        z = z.mean(dim=0)  
        return z.cpu().numpy()



def calculate_similarity(emb1, emb2):

    sims = []
    for a, b in zip(emb1, emb2):
        s = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
        sims.append(s)

    return float(np.mean(sims)), sims

@dataclass
class GraphMetrics:
    
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    similarity: float = 0.0


class GraphEvaluator:
    def __init__(self, model=None):
        self.results = {}
        self.model = model
    def extract_graph_elements(self, data: List[Dict]) -> Dict[str, Dict]:

        graphs = {}

        for doc in data:
            doc_key = doc["doc_key"]

            entities = set()
            if "ners" in doc:
                for ner in doc["ners"]:
               
                    entities.add((ner["text"].lower(), ner["type"].lower()))

         
            relations = set()
            if "relations" in doc:
                for rel in doc["relations"]:
                    if len(rel) >= 5:
                        relations.add((
                            rel[0].lower(),
                            rel[1].lower(),
                            rel[2].lower(),
                            rel[3].lower(),
                            rel[4].lower()
                        ))

            graphs[doc_key] = {
                "entities": entities,
                "relations": relations
            }

        return graphs

    def compute_graph_similarity(self, pred_doc, gt_doc):
 
        pred_data = to_pyg_data(pred_doc)
        gt_data = to_pyg_data(gt_doc)

        emb_pred = get_graph_embedding(self.model, pred_data)
        emb_gt = get_graph_embedding(self.model, gt_data)

        return cosine_similarity(emb_pred.reshape(1, -1), emb_gt.reshape(1, -1))[0][0]

    def calculate_similarity2(self, set1: Set, set2: Set) -> float:
    
        if not set1 and not set2:
            return 1.0
        elif not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    def calculate_metrics(self, predicted: Set, ground_truth: Set) -> Tuple[float, float, float]:
     
        if not predicted and not ground_truth:
            return 0.0, 0.0, 0.0

        tp = len(predicted.intersection(ground_truth))
        fp = len(predicted - ground_truth)
        fn = len(ground_truth - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def evaluate_graph_level(self, pred_graphs: Dict, gt_graphs: Dict,
                             pred_docs_raw: Dict, gt_docs_raw: Dict) -> Dict[str, GraphMetrics]:

    
        metrics_dict = {}

        all_docs = set(pred_graphs.keys()).union(set(gt_graphs.keys()))

        
        total_entity_metrics = np.zeros(3)  # precision, recall, f1
        total_relation_metrics = np.zeros(3)  # precision, recall, f1
        total_similarity = 0.0
        count = 0

        for doc_key in all_docs:
            pred = pred_graphs.get(doc_key, {"entities": set(), "relations": set()})
            gt = gt_graphs.get(doc_key, {"entities": set(), "relations": set()})

            
            entity_precision, entity_recall, entity_f1 = self.calculate_metrics(
                pred["entities"], gt["entities"]
            )

           
            relation_precision, relation_recall, relation_f1 = self.calculate_metrics(
                pred["relations"], gt["relations"]
            )

            graph_similarity = 0.0
            if self.model is not None and doc_key in pred_docs_raw and doc_key in gt_docs_raw:
                graph_similarity = self.compute_graph_similarity(pred_doc=pred_docs_raw[doc_key],
                                                                 gt_doc=gt_docs_raw[doc_key])

            
            entity_sim = self.calculate_similarity2(pred["entities"], gt["entities"])
            relation_sim = self.calculate_similarity2(pred["relations"], gt["relations"])
            graph_similarity = (graph_similarity + (entity_sim + relation_sim) / 2 )/2

           
            avg_precision = (entity_precision + relation_precision) / 2
            avg_recall = (entity_recall + relation_recall) / 2
            avg_f1 = (entity_f1 + relation_f1) / 2

            metrics = GraphMetrics(
                precision=avg_precision,
                recall=avg_recall,
                f1=avg_f1,
                similarity=graph_similarity
            )
            metrics_dict[doc_key] = metrics

            
            total_entity_metrics += [entity_precision, entity_recall, entity_f1]
            total_relation_metrics += [relation_precision, relation_recall, relation_f1]
            total_similarity += graph_similarity
            count += 1

        
        if count > 0:
            avg_entity_metrics = total_entity_metrics / count
            avg_relation_metrics = total_relation_metrics / count
            avg_similarity = total_similarity / count
            avg_f1 = (avg_entity_metrics[2] + avg_relation_metrics[2]) / 2
            avg_precision = (avg_entity_metrics[0] + avg_relation_metrics[0]) / 2
            avg_recall = (avg_entity_metrics[1] + avg_relation_metrics[1]) / 2

            metrics_dict["overall"] = GraphMetrics(
                precision=avg_precision,
                recall=avg_recall,
                f1=avg_f1,
                similarity=avg_similarity
            )

         
            metrics_dict["entity_average"] = GraphMetrics(
                precision=avg_entity_metrics[0],
                recall=avg_entity_metrics[1],
                f1=avg_entity_metrics[2],
                similarity=0.0
            )

            metrics_dict["relation_average"] = GraphMetrics(
                precision=avg_relation_metrics[0],
                recall=avg_relation_metrics[1],
                f1=avg_relation_metrics[2],
                similarity=0.0
            )

        return metrics_dict


    def save_results(self, metrics_dict: Dict[str, GraphMetrics], output_file: str):

        results_dict = {}
        for doc_key, metrics in metrics_dict.items():
            results_dict[doc_key] = {
                "precision": float(metrics.precision),
                "recall": float(metrics.recall),
                "f1": float(metrics.f1),
                "similarity": float(metrics.similarity)
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

    


def load_json_file(filepath: str) -> List[Dict]:
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gt_file', type=str, default='ag/AG_gt.json')
    parser.add_argument('--pred_file', type=str, default='ag/AG_ClarityG.json')
    parser.add_argument('--output', type=str, default='graph_evaluation_results.json')
    args = parser.parse_args()

    gt_data = load_json_file(args.gt_file)
    pred_data = load_json_file(args.pred_file)

    all_docs = gt_data
    all_graph_data = [to_pyg_data(doc) for doc in all_docs]
    feature_dim = len({"process", "file", "socket"})
  
    model = train_autoencoder(all_graph_data, feature_dim)


    evaluator = GraphEvaluator(model=model)

    gt_graphs = evaluator.extract_graph_elements(gt_data)
    pred_graphs = evaluator.extract_graph_elements(pred_data)

    gt_docs_raw = {doc['doc_key']: doc for doc in gt_data}
    pred_docs_raw = {doc['doc_key']: doc for doc in pred_data}

    metrics_dict = evaluator.evaluate_graph_level(pred_graphs, gt_graphs, pred_docs_raw, gt_docs_raw)

    evaluator.save_results(metrics_dict, args.output)


if __name__ == "__main__":
    main()

'''
python CRUcialG_extractor_compare.py --gt_file ag/AG_gt.json --pred_file ag/AG_ClarityG.json --output AG_ClarityG_results.json
'''

