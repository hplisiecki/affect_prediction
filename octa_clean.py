import numpy as np
import scipy.stats as st
import random
import pandas as pd
from tqdm import tqdm
import xlrd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import GridSearchCV
from matplotlib import animation
from IPython.display import HTML
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from scipy.spatial import Delaunay, ConvexHull, KDTree
from transformers import PreTrainedTokenizerFast, RobertaModel
import os

###############################################################################
"""
Octa without standard deviations
"""
###############################################################################


import wandb

words_full = pd.read_excel("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4947584/bin/DataSheet1.XLSX", sheet_name="Arkusz1", index_col = 0)
words = words_full.loc[:,[col for col in words_full.columns if not ("Male" in col or "Female" in col or
                                                          "MIN" in col or "MAX" in col or "_N" in col)]]

words = words.rename(columns={"part of speach":"part of speech"}) # Poprawka miss spellingu
print(words.columns) #Jakie mamy informacje
words.head()
words = words.set_index("polish word")
ratings = words.loc[:,[col for col in words.columns if "M" in col or "part of speech" in col or 'SD' in col]]# Wybieranie samych średnich ocen

# colors = ratings["valence"]
ratings = ratings.drop("part of speech", axis=1)
ratings_org = ratings.copy()  #Oryignalne oceny przed normalizacją
ratings = (ratings-ratings.mean())/ratings.std() #normalize ratings
ratings_nouns = ratings.loc[words["part of speech"]=="N",:] #Same rzeczowniki

ratings.reset_index(level=0, inplace=True)



ratings['norm_valence'] = (ratings['Valence_M'] - min(ratings['Valence_M'])) / (max(ratings['Valence_M']) - min(ratings['Valence_M']))
ratings['norm_arousal'] = (ratings['arousal_M'] - min(ratings['arousal_M'])) / (max(ratings['arousal_M']) - min(ratings['arousal_M']))
ratings['norm_dominance'] = (ratings['dominance_M'] - min(ratings['dominance_M'])) / (max(ratings['dominance_M']) - min(ratings['dominance_M']))
ratings['norm_origin'] = (ratings['origin_M'] - min(ratings['origin_M'])) / (max(ratings['origin_M']) - min(ratings['origin_M']))
ratings['norm_significance'] = (ratings['significance_M'] - min(ratings['significance_M'])) / (max(ratings['significance_M']) - min(ratings['significance_M']))
ratings['norm_concretness'] = (ratings['concretness_M'] - min(ratings['concretness_M'])) / (max(ratings['concretness_M']) - min(ratings['concretness_M']))
ratings['norm_imegability'] = (ratings['imegability_M'] - min(ratings['imegability_M'])) / (max(ratings['imegability_M']) - min(ratings['imegability_M']))
ratings['norm_age_of_aquisition'] = (ratings['ageOfAquisition_M'] - min(ratings['ageOfAquisition_M'])) / (max(ratings['ageOfAquisition_M']) - min(ratings['ageOfAquisition_M']))


# print("MAE przewiwydania średniej Valence dla ANEW: ",(ratings['Valence Mean']-ratings['Valence Mean'].mean()).abs().mean())

import torch
import numpy as np
from transformers import BertTokenizer



model_dir = "C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
# add pad token
tokenizer.pad_token = 0


# t = [tokenizer(str(text),
#                                padding='max_length', max_length = 20, truncation=True,
#                                 return_tensors="pt") for text in ratings['polish word']]
# longest = 0
# for i in t:
#     mask = i['attention_mask']
#     if mask.sum() > longest:
#         longest = mask.sum()
# print(longest)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels_valence = df['norm_valence'].values.astype(float)
        self.labels_arousal = df['norm_arousal'].values.astype(float)
        self.labels_dominance = df['norm_dominance'].values.astype(float)
        self.labels_origin = df['norm_origin'].values.astype(float)
        self.labels_significance = df['norm_significance'].values.astype(float)
        self.labels_concretness = df['norm_concretness'].values.astype(float)
        self.labels_imegability = df['norm_imegability'].values.astype(float)
        self.labels_age_of_aquisition = df['norm_age_of_aquisition'].values.astype(float)

        self.texts = [tokenizer(str(text),
                               padding='max_length', max_length = 10, truncation=True,
                                return_tensors="pt") for text in df['polish word']]

    def classes(self):
        return self.labels_valence, self.labels_arousal, self.labels_dominance, self.labels_origin, self.labels_significance, self.labels_concretness, self.labels_imegability, self.labels_age_of_aquisition, self.labels_valencesd

    def __len__(self):
        return len(self.labels_valence)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels_valence[idx]), np.array(self.labels_arousal[idx]), np.array(self.labels_dominance[idx]), np.array(self.labels_origin[idx]), np.array(self.labels_significance[idx]), np.array(self.labels_concretness[idx]), np.array(self.labels_imegability[idx]), np.array(self.labels_age_of_aquisition[idx])

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
# df_train.to_csv('train_octa_clean.csv', index=False)
# df_val.to_csv('val_octa_clean.csv', index=False)
# df_test.to_csv('test_octa_clean.csv', index=False)

df_train = pd.read_csv('train_octa_clean.csv')
df_val = pd.read_csv('val_octa_clean.csv')
df_test = pd.read_csv('test_octa_clean.csv')

model_dir = "C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/"

class BertRegression(nn.Module):

    def __init__(self, dropout=0.2, hidden_dim=768):

        super(BertRegression, self).__init__()

        self.bert = RobertaModel.from_pretrained(model_dir)
        self.affect = nn.Linear(hidden_dim, 1)
        self.arousal = nn.Linear(hidden_dim, 1)
        self.dominance = nn.Linear(hidden_dim, 1)
        self.origin = nn.Linear(hidden_dim, 1)
        self.significance = nn.Linear(hidden_dim, 1)
        self.concreteness = nn.Linear(hidden_dim, 1)
        self.imageability = nn.Linear(hidden_dim, 1)
        self.aqcuisition = nn.Linear(hidden_dim, 1)

        self.l_1_affect = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_arousal = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_dominance = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_origin = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_significance = nn.Linear(hidden_dim, hidden_dim)
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

        affect_all = self.relu(self.dropout(self.layer_norm(self.l_1_affect(x) + x)))
        arousal_all = self.relu(self.dropout(self.layer_norm(self.l_1_arousal(x) + x)))
        dominance_all = self.relu(self.dropout(self.layer_norm(self.l_1_dominance(x) + x)))
        origin_all = self.relu(self.dropout(self.layer_norm(self.l_1_origin(x) + x)))
        significance_all = self.relu(self.dropout(self.layer_norm(self.l_1_significance(x) + x)))
        concreteness_all = self.relu(self.dropout(self.layer_norm(self.l_1_concreteness(x) + x)))
        imageability_all = self.relu(self.dropout(self.layer_norm(self.l_1_imageability(x) + x)))
        aqcuisition_all = self.relu(self.dropout(self.layer_norm(self.l_1_aqcuisition(x) + x)))



        affect = self.sigmoid(self.affect(affect_all))
        arousal = self.sigmoid(self.arousal(arousal_all))
        dominance = self.sigmoid(self.dominance(dominance_all))
        origin = self.sigmoid(self.origin(origin_all))
        significance = self.sigmoid(self.significance(significance_all))
        concreteness = self.sigmoid(self.concreteness(concreteness_all))
        imageability = self.sigmoid(self.imageability(imageability_all))
        aqcuisition = self.sigmoid(self.aqcuisition(aqcuisition_all))



        return affect, arousal, dominance, origin, significance, concreteness, imageability, aqcuisition



from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

epochs = 1000
model = BertRegression()





train, val = Dataset(df_train), Dataset(df_val)


batch_size = 500
train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# import torch.nn.MSELoss
# criterion = torch.nn.L1Loss()
# cross entropy loss
criterion = torch.nn.MSELoss()
# optimizer = Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=5e-4,
                  eps=1e-8,
                  weight_decay=0.3,
                  amsgrad=True,
                  betas = (0.9, 0.999))

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=600,
                                            num_training_steps=len(train_dataloader) * epochs)

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

wandb.init(project="dutch", entity="hubertp")
wandb.watch(model, log_freq=5)

# for param in model.bert.parameters():
#     print(param.requires_grad)
#     param.requires_grad = False

h = 1
best_loss = 100
for epoch_num in range(epochs):
    total_loss_train = 0

    for train_input, (v, a, d, o, s, c, i, aq) in tqdm(train_dataloader):
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)

        train_label = torch.cat((v, a, d, o, s, c, i, aq), dim=0).to(device)
        del v, a, d, o, s, c, i, aq

        o1, o2, o3, o4, o5, o6, o7, o8 = model(input_id, mask)
        del input_id, mask
        # concatenate
        output_a = torch.cat((o1, o2, o3, o4, o5, o6, o7, o8), dim=0)
        del o1, o2, o3, o4, o5, o6, o7, o8

        l1 = criterion(output_a.float(), train_label.view(-1,1).float())
        del output_a, train_label


        batch_loss = l1
        total_loss_train += batch_loss.item()



        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
    total_acc_val = 0
    total_loss_val = 0

    with torch.no_grad():

        for val_input, (val_v, val_a, val_d, val_o, val_s, val_c, val_i, val_aq) in val_dataloader:
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)

            val_label = torch.cat((val_v, val_a, val_d, val_o, val_s, val_c, val_i, val_aq), dim=0).to(device)
            del val_v, val_a, val_d, val_o, val_s, val_c, val_i, val_aq




            o1, o2, o3, o4, o5, o6, o7, o8 = model(input_id, mask)
            del input_id, mask

            val_output_a = torch.cat((o1, o2, o3, o4, o5, o6, o7, o8), dim=0)
            del o1, o2, o3, o4, o5, o6, o7, o8

            l1 = criterion(val_output_a.float(), val_label.view(-1,1).float())
            del val_label, val_output_a


            batch_loss = l1
            total_loss_val += batch_loss.item()
        if total_loss_val < best_loss:
            best_loss = total_loss_val
            torch.save(model.state_dict(), './models/octa_clean.pt')

    if epoch_num % 2 == 0:
        wandb.log({"loss": total_loss_train / len(df_train), "lr": scheduler.get_last_lr()[0], "epoch": epoch_num, "val_loss": total_loss_val/ len(df_val)})
    print(
        f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(df_train): .10f} \
            | Val Loss: {total_loss_val / len(df_val): .10f}')

# load
model.load_state_dict(torch.load('./models/octa_clean.pt'))

def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        preval, prearo, predom, preori, presig, precon, preim, preage, trueval, truearo, truedom, trueori, truesig, truecon, trueim, trueage = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for test_input, (test_valence, test_arousal, test_dominance, test_origin, test_significance, test_concretness, test_imageability, test_acquisition) in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            o1, o2, o3, o4, o5, o6, o7, o8 = model(input_id, mask)
            # batch_loss = criterion(output.float(), val_label.float())


            preval.extend([p for p in o1.cpu()])
            prearo.extend([p for p in o2.cpu()])
            predom.extend([p for p in o3.cpu()])
            preori.extend([p for p in o4.cpu()])
            presig.extend([p for p in o5.cpu()])
            precon.extend([p for p in o6.cpu()])
            preim.extend([p for p in o7.cpu()])
            preage.extend([p for p in o8.cpu()])


            trueval.extend([t for t in test_valence.cpu()])
            truearo.extend([t for t in test_arousal.cpu()])
            truedom.extend([t for t in test_dominance.cpu()])
            trueori.extend([t for t in test_origin.cpu()])
            truesig.extend([t for t in test_significance.cpu()])
            truecon.extend([t for t in test_concretness.cpu()])
            trueim.extend([t for t in test_imageability.cpu()])
            trueage.extend([t for t in test_acquisition.cpu()])

            # print loss
    return preval, prearo, predom , preori, presig, precon, preim, preage, trueval, truearo, truedom, trueori, truesig, truecon, trueim, trueage

pred_val, pred_aro, pred_dom, pred_ori, pred_sig, pred_con, pred_im, pred_age, true_val, true_aro, true_dom, true_ori, true_sig, true_con, true_im, true_age = evaluate(model, df_test)


def get_diffs_and_corr(preds, trues):
    diffs = []
    for i in range(len(preds)):
        diffs.append(abs(preds[i] - trues[i]))
    mean = sum(diffs) / len(diffs)
    p = [float(i) for i in preds]
    t = [float(i) for i in trues]
    corr = np.corrcoef(p, t)[0][1]
    return mean, corr


mean_val, corr_val = get_diffs_and_corr(pred_val, true_val)
mean_aro, corr_aro = get_diffs_and_corr(pred_aro, true_aro)
mean_dom, corr_dom = get_diffs_and_corr(pred_dom, true_dom)
mean_ori, corr_ori = get_diffs_and_corr(pred_ori, true_ori)
mean_sig, corr_sig = get_diffs_and_corr(pred_sig, true_sig)
mean_con, corr_con = get_diffs_and_corr(pred_con, true_con)
mean_im, corr_im = get_diffs_and_corr(pred_im, true_im)
mean_age, corr_age = get_diffs_and_corr(pred_age, true_age)

print('valence: ' + str(corr_val))
print('arousal: ' + str(corr_aro))
print('dominance: ' + str(corr_dom))
print('origin: ' +  str(corr_ori))
print('significance: ' + str(corr_sig))
print('concretness: ' + str(corr_con))
print('imageability: ' + str(corr_im))
print('age of acquisition: ' + str(corr_age))

print('valence: ' + str(mean_val))
print('arousal: ' + str(mean_aro))
print('dominance: ' + str(mean_dom))
print('origin: ' +  str(mean_ori))
print('significance: ' + str(mean_sig))
print('concretness: ' + str(mean_con))
print('imageability: ' + str(mean_im))
print('age of acquisition: ' + str(mean_age))