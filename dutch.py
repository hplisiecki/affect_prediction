import pandas as pd
import wandb
import torch
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, AutoTokenizer, RobertaModel, AutoModel
from repo.training_loop import training_loop
from repo.dataset_and_model import Dataset, BertRegression
from transformers import logging
logging.set_verbosity_error()

###############################################################################
"""
Dutch word norm bert training script
"""
###############################################################################
###############################################################################
# HYPERPARAMETERS
#################################
# 7, 8
max_len = [7, 8]
max_len = 8
hidden_dim = 1536
hidden_dim = 768
dropout = 0.1
warmup_steps = 600
save_dir = 'models/gronlp_ducth_base.pth'
#
metric_names = ['valence', 'arousal', 'dominance', 'aqcuisition']

model_dir2 = "GroNLP/bert-base-dutch-cased"

model_name = ['bert2']

model_initialization = [RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base"),
                       AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")]
model_initialization = [RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base")]
model_initialization = [AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")]
epochs = 1000
batch_size = 100
learning_rate = 5e-5
eps = 1e-8
weight_decay = 0.3
amsgrad = True
betas = (0.9, 0.999)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

###############################################################################
# DATA LOADING
#################################

df_train = pd.read_parquet('data/train_dutch.parquet')
df_val = pd.read_parquet('data/val_dutch.parquet')
df_test = pd.read_parquet('data/test_dutch.parquet')

###############################################################################
# INITIALIZATION
#################################

# TOKENIZERS
tokenizer = AutoTokenizer.from_pretrained(model_dir2)
tokenizer = tokenizer

# MODEL
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)

# DATALOADERS
train, val = Dataset(tokenizer, df_train, max_len, metric_names), Dataset(tokenizer, df_val, max_len, metric_names)
train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

# TRAINING SETTINGS
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=learning_rate, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, betas=betas)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = len(train_dataloader) * epochs)

###############################################################################
# TRAINING
#################################

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

wandb.init(project="dutch", entity="hubertp")
wandb.watch(model, log_freq=5)

training_loop(model, optimizer, scheduler, epochs, train_dataloader, val_dataloader, criterion,
              device, save_dir, use_wandb = True)