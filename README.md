# ClarityG

**ClarityG: Enhancing CTI-Report-Driven Attack Graph Construction with Command Line Evidence**

ClarityG is an attack graph reconstruction framework for Cyber Threat Intelligence (CTI). It automatically identifies high-value command-line evidence from CTI reports and integrates it into attack graphs, enabling a more complete and fine-grained representation of adversarial behaviors.

------

## Environment & Dependencies

Python **3.8** is recommended.

```
pip install \
torch==2.4.1 \
transformers==4.46.3 \
accelerate==1.0.1 \
sentence-transformers==3.2.1 \
scikit-learn==1.3.2 \
numpy==1.24.4 \
scipy==1.10.1 \
faiss-cpu==1.8.0.post1 \
networkx==3.1 \
graphviz==0.20.3 \
torch-geometric==2.6.1 \
bashlex==0.18 \
regex==2024.11.6 \
pandas==2.0.3 \
matplotlib==3.7.5
```

------

## Usage

### Command Line / Text Classification

#### Train the classifier

```
python classifier_cmd_text.py
```

#### Identify command lines using the trained model

```
python comRecognition.py
```

------

### Initial Attack Graph Construction

Construct the attack graph based on the outputs of **CRUCialG**:

```
python ASG_gengerator.py \
  --graph_generator_json experiment/ag/crucialg/test_result_all.json \
  --asg_reconstruction_json experiment/ag/crucialg/test_AG_result_all.json \
  --asg_reconstruction_graph experiment/ag/crucialg/graph
```

------

### Command-Line Entity and Relation Extraction

#### Model training and evaluation

```
python main.py \
  --do_train \
  --do_eval \
  --task semeval \
  --data_dir ./data_CTI \
  --model_dir ./model \
  --eval_dir ./eval_CTI \
  --train_file train.tsv \
  --test_file test.tsv \
  --label_file label.txt \
  --model_name_or_path /root/roberta/R-BERT-master/bert-base-uncased
```

------

### Tactic and Technique Recognition

#### Command-line TTP recognition

```
python train_cmd_text.py
```

#### Text-based TTP recognition

```
python train_text_roberta.py
```

------

### Attack Graph Enhancement and Fusion

#### Generate an attack subgraph from a single command line

```
python cmd2Ag.py
```

#### Merge command-line attack subgraphs with CTI-derived attack graphs

```
python cmd_CTI_merge.py
```

