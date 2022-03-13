import pandas as pd
import wandb
import os
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch import nn
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# df = pd.read_excel('Redondo-BRM-2007/Redondo(2007).xls')
df = pd.read_excel('Redondo-BRM-2007/concaro.xlsx')

df['norm_valence'] = (df['VAL_M'] - min(df['VAL_M'])) / (max(df['VAL_M']) - min(df['VAL_M']))
df['norm_arousal'] = (df['ARO_M'] - min(df['ARO_M'])) / (max(df['ARO_M']) - min(df['ARO_M']))
df['norm_concreteness'] = (df['CON_M'] - min(df['CON_M'])) / (max(df['CON_M']) - min(df['CON_M']))
df['norm_imagebility'] = (df['IMA_M'] - min(df['IMA_M'])) / (max(df['IMA_M']) - min(df['IMA_M']))
df['norm_familiarity'] = (df['FAM_M'] - min(df['FAM_M'])) / (max(df['FAM_M']) - min(df['FAM_M']))

import torch
import numpy as np
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
# add pad token
tokenizer.pad_token = 0


# t = [tokenizer(str(text),
#                                padding='max_length', max_length = 100, truncation=True,
#                                 return_tensors="pt") for text in df['Word']]
# longest = 0
# for i in t:
#     mask = i['attention_mask']
#     if mask.sum() > longest:
#         longest = mask.sum()
# print(longest)


# Valence_M
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels_valence = df['norm_valence'].values.astype(float)
        self.labels_arousal = df['norm_arousal'].values.astype(float)
        self.labels_concreteness = df['norm_concreteness'].values.astype(float)
        self.labels_imagebility = df['norm_imagebility'].values.astype(float)
        self.labels_familiarity = df['norm_familiarity'].values.astype(float)


        self.texts = [tokenizer(str(text),
                               padding='max_length', max_length = 10, truncation=True,
                                return_tensors="pt") for text in df['polish word']]

    def classes(self):
        return self.labels_valence, self.labels_arousal, self.labels_concreteness, self.labels_imagebility, self.labels_familiarity
    def __len__(self):
        return len(self.labels_valence)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels_valence[idx]), np.array(self.labels_arousal[idx]), np.array(self.labels_concreteness[idx]), np.array(self.labels_imagebility[idx]), np.array(self.labels_familiarity[idx])
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y




from torch import nn
from transformers import BertModel

np.random.seed(112)
df_train, df_val, df_test = np.split(ratings.sample(frac=1, random_state=42),
                                     [int(.8*len(ratings)), int(.9*len(ratings))])


# save
# df_train.to_csv('train_spanish.csv', index=False)
# df_val.to_csv('val_spanish.csv', index=False)
# df_test.to_csv('test_spanish.csv', index=False)

df_train = pd.read_csv('train_spanish.csv')
df_val = pd.read_csv('val_spanish.csv')
df_test = pd.read_csv('test_spanish.csv')


model_dir = "C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/"

class BertRegression(nn.Module):

    def __init__(self, dropout=0.2, hidden_dim=768):

        super(BertRegression, self).__init__()

        self.bert = RobertaModel.from_pretrained(model_dir)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.valence = nn.Linear(hidden_dim, 1)
        self.arousal = nn.Linear(hidden_dim, 1)
        self.concreteness = nn.Linear(hidden_dim, 1)
        self.imageability = nn.Linear(hidden_dim, 1)
        self.familiarity = nn.Linear(hidden_dim, 1)

        self.val_sd = nn.Linear(hidden_dim, 1)
        self.ar_sd = nn.Linear(hidden_dim, 1)
        self.co_sd = nn.Linear(hidden_dim, 1)
        self.im_sd = nn.Linear(hidden_dim, 1)
        self.fam_sd = nn.Linear(hidden_dim, 1)

        self.l_1_valence = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_arousal = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_concreteness = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_imageability = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_aqcuisition = nn.Linear(hidden_dim, hidden_dim)



        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, x = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        x = self.dropout(x)

        valence_all = self.relu(self.dropout(self.layer_norm(self.l_1_valence(x) + x)))
        arousal_all = self.relu(self.dropout(self.layer_norm(self.l_1_arousal(x) + x)))
        concreteness_all = self.relu(self.dropout(self.layer_norm(self.l_1_concreteness(x) + x)))
        imageability_all = self.relu(self.dropout(self.layer_norm(self.l_1_imageability(x) + x)))
        familiarity_all = self.relu(self.dropout(self.layer_norm(self.l_1_familiarity(x) + x)))



        valence = self.sigmoid(self.valence(valence_all))
        arousal = self.sigmoid(self.arousal(arousal_all))
        concreteness = self.sigmoid(self.concreteness(concreteness_all))
        imageability = self.sigmoid(self.imageability(imageability_all))
        familiarity = self.sigmoid(self.aqcuisition(familiarity_all))



        return valence, arousal, concreteness, imageability, familiarity



epochs = 1000
model = BertRegression()


train, val = Dataset(df_train), Dataset(df_val)

batch_size = 100
train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion1 = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=5e-5,
                  eps=1e-8,  # Epsilon
                  weight_decay=0.3,
                  amsgrad=True,
                  betas = (0.9, 0.999))

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=600,
                                            num_training_steps=len(train_dataloader) * epochs)


if use_cuda:
    model = model.cuda()
    criterion1 = criterion1.cuda()




wandb.init(project="affect_anew", entity="hubertp")
wandb.watch(model, log_freq=5)

h = 0
best_loss = 150
for epoch_num in range(epochs):
    total_loss_train = 0

    for train_input1, train_input2, (valence, arousal, dominance, aoa, conc) in tqdm(train_dataloader):
        break
    break
        mask1 = train_input1['attention_mask'].to(device)
        input_id1 = train_input1['input_ids'].squeeze(1).to(device)
        mask2 = train_input2['attention_mask'].to(device)
        input_id2 = train_input2['input_ids'].squeeze(1).to(device)
        train_label = torch.cat((valence, arousal, dominance, aoa, conc), dim=0).to(device)
        output1, output2, output3, output4, output5 = model(input_id1, mask1, input_id2, mask2)

        del input_id1, mask1, input_id2, mask2

        output_a = torch.cat((output1, output2, output3, output4, output5), dim=0)

        l1 = criterion1(output_a.float(), train_label.view(-1,1).float())

        batch_loss = l1

        total_loss_train += batch_loss.item()
        model.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

    total_acc_val = 0
    total_loss_val = 0
    total_corr_valence = 0
    total_corr_arousal = 0
    total_corr_dominance = 0
    total_corr_aoa = 0
    total_corr_conc = 0
    total_corr = 0
    best_corr_valence = 0
    best_corr_arousal = 0
    best_corr_dominance = 0
    best_corr_aoa = 0
    best_corr_conc = 0
    with torch.no_grad():
        for val_input1, val_input2, (val_valence, val_arousal, val_dominance, val_aoa, val_conc) in val_dataloader:

            mask1 = val_input1['attention_mask'].to(device)
            input_id1 = val_input1['input_ids'].squeeze(1).to(device)
            mask2 = val_input2['attention_mask'].to(device)
            input_id2 = val_input2['input_ids'].squeeze(1).to(device)
            val_label = torch.cat((val_valence, val_arousal, val_dominance, val_aoa, val_conc), dim=0).to(device)
            output1, output2, output3, output4, output5 = model(input_id1, mask1, input_id2, mask2)

            val_output_a = torch.cat((output1, output2, output3, output4, output5), dim=0).to(device)
            l1 = criterion1(val_output_a.float(), val_label.view(-1,1).float())


            batch_loss = l1
            total_loss_val += batch_loss.item()

            output1, output2, output3, output4, output5 = output1.cpu().detach().view(-1).numpy(), output2.cpu().detach().view(-1).numpy(), output3.cpu().detach().view(-1).numpy(), output4.cpu().detach().view(-1).numpy(), output5.cpu().detach().view(-1).numpy()
            val_valence, val_arousal, val_dominance, val_aoa, val_conc = val_valence.numpy(), val_arousal.numpy(), val_dominance.numpy(), val_aoa.numpy(), val_conc.numpy()


            total_corr_valence += np.corrcoef(output1, val_valence)[0, 1]
            total_corr_arousal += np.corrcoef(output2, val_arousal)[0, 1]
            total_corr_dominance += np.corrcoef(output3, val_dominance)[0, 1]
            total_corr_aoa += np.corrcoef(output4, val_aoa)[0, 1]
            total_corr_conc += np.corrcoef(output5, val_conc)[0, 1]
            total_corr = (total_corr_valence + total_corr_arousal + total_corr_dominance) / 3

        # save best models
        if best_corr_valence / len(val_dataloader) < total_corr_valence / len(val_dataloader):
            best_corr_valence = total_corr_valence
            # delete the previous model
            if os.path.exists('models/best_corr_valence.pth'):
                os.remove('models/best_corr_valence.pth')
            torch.save(model.state_dict(), 'models/best_corr_valence.pth')

        if best_corr_arousal / len(val_dataloader) < total_corr_arousal / len(val_dataloader):
            best_corr_arousal = total_corr_arousal
            if os.path.exists('models/best_corr_arousal.pth'):
                os.remove('models/best_corr_arousal.pth')
            torch.save(model.state_dict(), 'models/best_corr_arousal.pth')

        if best_corr_dominance / len(val_dataloader) < total_corr_dominance / len(val_dataloader):
            best_corr_dominance = total_corr_dominance
            if os.path.exists('models/best_corr_dominance.pth'):
                os.remove('models/best_corr_dominance.pth')
            torch.save(model.state_dict(), 'models/best_corr_dominance.pth')

        if best_corr_aoa / len(val_dataloader) < total_corr_aoa / len(val_dataloader):
            best_corr_aoa = total_corr_aoa
            if os.path.exists('models/best_corr_aoa.pth'):
                os.remove('models/best_corr_aoa.pth')
            torch.save(model.state_dict(), 'models/best_corr_aoa.pth')

        if best_corr_conc / len(val_dataloader) < total_corr_conc / len(val_dataloader):
            best_corr_conc = total_corr_conc
            if os.path.exists('models/best_corr_conc.pth'):
                os.remove('models/best_corr_conc.pth')
            torch.save(model.state_dict(), 'models/best_corr_conc.pth')


    if epoch_num % 2 == 0:
        wandb.log({"loss": total_loss_train / len(df_train), "lr": scheduler.get_last_lr()[0], "epoch": epoch_num, "val_loss": total_loss_val/ len(df_val), "val_corr_valence": total_corr_valence / len(val_dataloader), "val_corr_arousal": total_corr_arousal / len(val_dataloader), "val_corr_dominance": total_corr_dominance / len(val_dataloader), 'val_corr_aoa': total_corr_aoa / len(val_dataloader), 'val_corr_conc': total_corr_conc / len(val_dataloader)})
    print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(df_train): .10f} \
            | Val Loss: {total_loss_val / len(df_val): .10f} | corr_valence: {total_corr_valence / len(val_dataloader): .10f} | corr_arousal: {total_corr_arousal / len(val_dataloader): .10f} | corr_dominance: {total_corr_dominance / len(val_dataloader): .10f} | corr_aoa: {total_corr_aoa / len(val_dataloader): .10f} | corr_conc: {total_corr_conc / len(val_dataloader): .10f}')

########################################################################################################################
########################################################################################################################
########################################################################################################################
