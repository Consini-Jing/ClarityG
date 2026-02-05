import os
import json
import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from torch.nn import Linear
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, ARGVA



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64         
HIDDEN_DIM = EMBED_DIM * 2
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 4



def load_json_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
    return g



def scan_graph_types(file_path):
    node_types = set()
    edge_types = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            for n in g["nodes"]:
                node_types.add(n["type"])
            for e in g["edges"]:
                edge_types.add(e["type"])
    node_types = sorted(list(node_types))
    edge_types = sorted(list(edge_types))
    node_map = {t: i for i, t in enumerate(node_types)}
    edge_map = {t: i for i, t in enumerate(edge_types)}

    return node_map, edge_map



def graph_to_data(g, node_map, edge_map):
    node_index = {}
    node_features = []

    for idx, node in enumerate(g["nodes"]):
        node_index[node["id"]] = idx
        node_features.append(node_map[node["type"]])

    x = torch.eye(len(node_map))[node_features]

    edge_index = []
    edge_attr = []

    for e in g["edges"]:
        src = node_index[e["source"]]
        tgt = node_index[e["target"]]
        etype = edge_map[e["type"]]
   
        edge_index.append([src, tgt])
        edge_index.append([tgt, src])
        edge_attr.append([etype])
        edge_attr.append([etype])


    num_nodes = len(node_features)
    if num_nodes <= 1:
     
        edge_index.append([0, 0])
        edge_index.append([0, 0])
        edge_attr.append([0])
        edge_attr.append([0])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)



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

      
            try:
                loss = model.recon_loss(z, data.edge_index)
            except AssertionError:
        
                loss = torch.tensor(0.0, device=device)

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
    sims = [(x + 1) / 2 for x in sims]
    return float(np.mean(sims)), sims



def load_graph_file(jsonl_path, node_map, edge_map):
    datalist = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            d = graph_to_data(g, node_map, edge_map)
            datalist.append(d)
    return datalist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_gt", type=str)
    parser.add_argument("--test_gt", type=str)
    parser.add_argument("--test_pred", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model", type=str, default="./graph_autoencoder.pt")
    args = parser.parse_args()

    node_map, edge_map = scan_graph_types(args.train_gt)

    train_list = load_graph_file(args.train_gt, node_map, edge_map)

    if args.eval:
        test_gt_list = load_graph_file(args.test_gt, node_map, edge_map)
        test_pred_list = load_graph_file(args.test_pred, node_map, edge_map)

    if args.train:
        print("==== Training Graph AutoEncoder ====")
        model = train_autoencoder(train_list, feature_dim=len(node_map))
        torch.save(model, args.model)
        

    if args.eval:
        print("==== Evaluating Similarity ====")
        model = torch.load(args.model)
        gt_emb = np.array([get_graph_embedding(model, g) for g in test_gt_list])
        pred_emb = np.array([get_graph_embedding(model, p) for p in test_pred_list])
        avg_sim, sims = calculate_similarity(gt_emb, pred_emb)
 

if __name__ == "__main__":
    main()

'''
python graph_similarity.py --train --train_gt data/train_gt.jsonl
python graph_similarity.py --train --train_gt data/cmd_graph_GT.jsonl

python graph_similarity.py --eval --train_gt data/train_gt.jsonl --test_gt data/test_gt.jsonl  --test_pred data/test_pred.jsonl
python graph_similarity.py --eval --train_gt data/cmd_graph_GT.jsonl --test_gt data/cmd_graph_GT.jsonl  --test_pred data/generated_graphs2.jsonl
'''
