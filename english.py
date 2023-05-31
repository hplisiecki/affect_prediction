import wandb
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
import torch
import pandas as pd
from repo.training_loop import training_loop
from repo.dataset_and_model import Dataset, BertRegression
from transformers import logging
logging.set_verbosity_error()

###############################################################################
"""
English word norm bert training script 
"""
###############################################################################
###############################################################################
# HYPERPARAMETERS
#################################

# max_len = [8,8]
max_len = 8
# hidden_dim = 1536
hidden_dim = 768
dropout = 0.1
warmup_steps = 600
save_dir = 'models/english_one_nghuyong'

metric_names = ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']


model_dir1 = "nghuyong/ernie-2.0-en"

model_name = ['bert']

model_initialization = [AutoModel.from_pretrained("nghuyong/ernie-2.0-en")]

epochs = 1000
batch_size = 500
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

df_train = pd.read_parquet('data/warriner_anew_train.parquet')
df_val = pd.read_parquet('data/warriner_anew_val.parquet')
df_test = pd.read_parquet('data/warriner_anew_test.parquet')

###############################################################################
# INITIALIZATION
#################################
# TOKENIZERS
tokenizer1 = AutoTokenizer.from_pretrained(model_dir1)
tokenizer = tokenizer1
# MODEL
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)

# DATALOADERS
train, val = Dataset(tokenizer, df_train, max_len, metric_names), Dataset(tokenizer, df_val, max_len, metric_names)
train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=learning_rate, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, betas=betas)


scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=len(train_dataloader) * epochs)

###############################################################################
# TRAINING
#################################

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

wandb.init(project="affect_anew", entity="hubertp")
wandb.watch(model, log_freq=5)

training_loop(model, optimizer, scheduler, epochs, train_dataloader, val_dataloader, criterion,
              device, save_dir,   use_wandb = True)