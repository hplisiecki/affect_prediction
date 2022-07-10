from repo.utils import evaluate, get_metrics
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, PreTrainedTokenizerFast, RobertaTokenizer, RobertaModel
from repo.dataset_and_model import BertRegression
import torch
import pandas as pd
from transformers import logging
import os
logging.set_verbosity_error()

###############################################################################
"""
Calculates the results for all languages
"""
###############################################################################
###############################################################################
# ENGLISH
#################################
print("English")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']
max_len = [8,8]
hidden_dim = 1536
dropout = 0.1
model_dir1 = "finiteautomata/bertweet-base-emotion-analysis"
model_dir2 = "nghuyong/ernie-2.0-en"
model_name = ['bert1', 'bert2']
model_initialization = [AutoModel.from_pretrained('finiteautomata/bertweet-base-emotion-analysis'),
                        AutoModel.from_pretrained('nghuyong/ernie-2.0-en')]

# DATA LOADING
df_test = pd.read_parquet('warriner_anew_test.parquet')

# INITIALIZATION
tokenizer1 = AutoTokenizer.from_pretrained(model_dir1)
tokenizer2 = AutoTokenizer.from_pretrained(model_dir2)
tokenizer = [tokenizer1, tokenizer2]
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/english_no_drop'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)


###############################################################################
# DUTCH
#################################
print("Dutch")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aqcuisition']
max_len = [7, 8]
hidden_dim = 1536
dropout = 0.1
model_dir1 = "pdelobelle/robbert-v2-dutch-base"
model_dir2 = "GroNLP/bert-base-dutch-cased"
model_name = ['bert1', 'bert2']
model_initialization = [RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base"),
                       AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")]

# DATA LOADING
df_test = pd.read_parquet('test_dutch.parquet')

# INITIALIZATION
tokenizer1 = RobertaTokenizer.from_pretrained(model_dir1)
tokenizer1.pad_token = 0
tokenizer2 = AutoTokenizer.from_pretrained(model_dir2)
tokenizer = [tokenizer1, tokenizer2]
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('./models/dutch.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

###############################################################################
# SPANISH
#################################
print('Spanish')

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'concreteness', 'imageability', 'familiarity']
max_len = 6
hidden_dim = 768
dropout = 0.2
model_dir = 'dccuchile/bert-base-spanish-wwm-cased'
model_name = ['bert1']
model_initialization = [BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')]

# DATA LOADING
df_test = pd.read_parquet('test_spanish.parquet')

# INITIALIZATION
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/spanish.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

###############################################################################
# POLISH
#################################
print("Polish")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'origin', 'significance', 'concreteness', 'imageability', 'aqcuisition']
max_len = 10
hidden_dim = 768
dropout = 0.2
model_dir = "C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/"
model_name = ["bert"]
model_initialization = [RobertaModel.from_pretrained('C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/')]

# DATA LOADING
df_test = pd.read_parquet('test_octa_clean.parquet')

# INITIALIZATION
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
tokenizer.pad_token = 0
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/octa_clean.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)