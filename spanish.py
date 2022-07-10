import pandas as pd
import wandb
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel
from dataset_and_model import Dataset, BertRegression
import torch
from training_loop import training_loop
from transformers import logging
logging.set_verbosity_error()

###############################################################################
"""
Spanish word norm bert training script
"""
###############################################################################
###############################################################################
# HYPERPARAMETERS
#################################

max_len = 6
hidden_dim = 768
dropout = 0.2
warmup_steps = 600
save_dir = 'models/test_run'

metric_names = ['valence', 'arousal', 'concreteness', 'imageability', 'familiarity']

model_dir = 'dccuchile/bert-base-spanish-wwm-cased'

model_name = ['bert1']
model_initialization = [BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')]

epochs = 1000
batch_size = 300
learning_rate = 5e-4
eps = 1e-8
weight_decay = 0.3
amsgrad = True
betas = (0.9, 0.999)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

###############################################################################
# DATA LOADING
#################################

df_train = pd.read_parquet('train_spanish.parquet')
df_val = pd.read_parquet('val_spanish.parquet')
df_test = pd.read_parquet('test_spanish.parquet')

###############################################################################
# INITIALIZATION
#################################

# TOKENIZER
tokenizer = BertTokenizer.from_pretrained(model_dir)

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
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=len(train_dataloader) * epochs)

###############################################################################
# TRAINING
#################################

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

wandb.init(project="spanish", entity="hubertp")
wandb.watch(model, log_freq=5)

# LOOP
training_loop(model, optimizer, scheduler, epochs, train_dataloader, val_dataloader, criterion,
              device, save_dir, use_wandb = True)