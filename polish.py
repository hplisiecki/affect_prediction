import pandas as pd
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup, RobertaModel
import os
import wandb
import torch
from dataset_and_model import Dataset, BertRegression
from training_loop import training_loop

###############################################################################
"""
Polish word norm bert training script
"""
###############################################################################
###############################################################################
# HYPERPARAMETERS
#################################

max_len = 10
hidden_dim = 768
dropout = 0.2
warmup_steps = 600
save_dir = 'models/test_run'

metric_names = ["valence","arousal","dominance","origin","significance","concretness","imegability","age_of_aquisition"]

model_dir = "C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/"

model_name = ["bert"]
model_initialization = [RobertaModel.from_pretrained('C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/')]

epochs = 1000
batch_size = 500
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

df_train = pd.read_parquet('train_octa_clean.parquet')
df_val = pd.read_parquet('val_octa_clean.parquet')
df_test = pd.read_parquet('test_octa_clean.parquet')

###############################################################################
# INITIALIZATION
#################################

# TOKENIZER
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
tokenizer.pad_token = 0

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

wandb.init(project="dutch", entity="hubertp")
wandb.watch(model, log_freq=5)

# LOOP
training_loop(model, optimizer, scheduler, epochs, train_dataloader, val_dataloader, criterion,
              device, save_dir, use_wandb = True)
