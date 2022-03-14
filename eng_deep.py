import pandas as pd
import wandb
import os
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch import nn
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

warriner = pd.read_csv('C:/Users/hplis/OneDrive/Desktop/Koła/open_ai/Ratings_Warriner_et_al.csv')
bradley = pd.read_csv('https://raw.githubusercontent.com/mileszim/anew_formats/master/csv/all.csv')

a = warriner[warriner.Word == 'rape']

aoa = pd.read_excel('AoA.xlsx')
aoa['norm_aoa'] = (aoa['Rating.Mean'] - min(aoa['Rating.Mean'])) / (max(aoa['Rating.Mean']) - min(aoa['Rating.Mean']))
aoa = aoa.dropna(subset=['norm_aoa'])

esm = pd.read_excel('ESM.xlsx')
esm['norm_concreteness'] =  (esm['Conc.M'] - min(esm['Conc.M'])) / (max(esm['Conc.M']) - min(esm['Conc.M']))
esm = esm.dropna(subset=['norm_concreteness'])


model1 = "finiteautomata/bertweet-base-emotion-analysis"
model2 = "nghuyong/ernie-2.0-en"

# delete nan in "Words"
warriner = warriner.dropna(subset=['Word'])

# import sent_tokenize
# from nltk.tokenize import sent_tokenize
# import nltk
#
# type = []
# for word in tqdm(warriner['Word']):
#     w = sent_tokenize(word)
#     tagged = nltk.pos_tag(w)
#     type.append(tagged[0][1])
#
# warriner['type'] = type


# reformat to 0 to 1
warriner['norm_valence'] = (warriner['V.Mean.Sum'] - min(warriner['V.Mean.Sum'])) / (max(warriner['V.Mean.Sum']) - min(warriner['V.Mean.Sum']))
warriner['norm_arousal'] = (warriner['A.Mean.Sum'] - min(warriner['A.Mean.Sum'])) / (max(warriner['A.Mean.Sum']) - min(warriner['A.Mean.Sum']))
warriner['norm_dominance'] = (warriner['D.Mean.Sum'] - min(warriner['D.Mean.Sum'])) / (max(warriner['D.Mean.Sum']) - min(warriner['D.Mean.Sum']))

# sort
warriner['word_number'] = warriner['Word'].apply(lambda x: len(str(x).split()))
warriner['norm_word_number'] = (warriner['word_number'] - min(warriner['word_number'])) / (max(warriner['word_number']) - min(warriner['word_number']))

warriner['length'] = warriner['Word'].apply(lambda x: len(str(x)))

warriner['norm_length'] = (warriner['length'] - min(warriner['length'])) / (max(warriner['length']) - min(warriner['length']))

warriner['norm_dominance_sd'] = (warriner['D.SD.Sum'] - min(warriner['D.SD.Sum'])) / (max(warriner['D.SD.Sum']) - min(warriner['D.SD.Sum']))
warriner['norm_arousal_sd'] = (warriner['A.SD.Sum'] - min(warriner['A.SD.Sum'])) / (max(warriner['A.SD.Sum']) - min(warriner['A.SD.Sum']))
warriner['norm_valence_sd'] = (warriner['V.SD.Sum'] - min(warriner['V.SD.Sum'])) / (max(warriner['V.SD.Sum']) - min(warriner['V.SD.Sum']))

warriner['norm_arousal_n'] = (warriner['V.Rat.Sum'] - min(warriner['V.Rat.Sum'])) / (max(warriner['V.Rat.Sum']) - min(warriner['V.Rat.Sum']))
warriner['norm_valence_n'] = (warriner['A.Rat.Sum'] - min(warriner['A.Rat.Sum'])) / (max(warriner['A.Rat.Sum']) - min(warriner['A.Rat.Sum']))
warriner['norm_dominance_n'] = (warriner['D.Rat.Sum'] - min(warriner['D.Rat.Sum'])) / (max(warriner['D.Rat.Sum']) - min(warriner['D.Rat.Sum']))

warriner['var_valence'] = warriner['V.SD.Sum'] / np.sqrt(warriner['V.Rat.Sum'])
warriner['var_arousal'] = warriner['A.SD.Sum'] / np.sqrt(warriner['A.Rat.Sum'])
warriner['var_dominance'] = warriner['D.SD.Sum'] / np.sqrt(warriner['D.Rat.Sum'])




np.random.seed(112)

common = [x for x in list(warriner.Word.values) if x in list(bradley.Description.values)]
df_test = warriner[warriner.Word.isin(common)]
warriner = warriner[~warriner.Word.isin(common)]

df_train = warriner.sample(frac=0.9, random_state=42)
df_val = warriner.drop(df_train.index)

# save
# df_train.to_csv('warriner_anew_train.csv', index=False)
# df_test.to_csv('warriner_anew_test.csv', index=False)
# df_val.to_csv('warriner_anew_val.csv', index=False)

# append AoA and Concreteness scores to the previously generated train/val/test sets
df_train = pd.read_csv('warriner_anew_train.csv')

temp = aoa[aoa['Word'].isin(list(df_train['Word']))]
a = df_train[df_train['Word'].isin(list(temp['Word']))]
aoa_scores = [temp[temp['Word'] == w]['norm_aoa'].values[0] for w in list(a['Word'])]
a['norm_aoa'] = aoa_scores

temp = esm[esm['Word'].isin(list(a['Word']))]
df_train = a[a['Word'].isin(list(temp['Word']))]
concreteness_scores = [temp[temp['Word'] == w]['norm_concreteness'].values[0] for w in list(df_train['Word'])]
df_train['norm_concreteness'] = concreteness_scores


df_test = pd.read_csv('warriner_anew_test.csv')

temp = aoa[aoa['Word'].isin(list(df_test['Word']))]
a = df_test[df_test['Word'].isin(list(temp['Word']))]
aoa_scores = [temp[temp['Word'] == w]['norm_aoa'].values[0] for w in list(a['Word'])]
a['norm_aoa'] = aoa_scores

temp = esm[esm['Word'].isin(list(a['Word']))]
df_test = a[a['Word'].isin(list(temp['Word']))]
concreteness_scores = [temp[temp['Word'] == w]['norm_concreteness'].values[0] for w in list(df_test['Word'])]
df_test['norm_concreteness'] = concreteness_scores


df_val = pd.read_csv('warriner_anew_val.csv')

temp = aoa[aoa['Word'].isin(list(df_val['Word']))]
a = df_val[df_val['Word'].isin(list(temp['Word']))]
aoa_scores = [temp[temp['Word'] == w]['norm_aoa'].values[0] for w in list(a['Word'])]
a['norm_aoa'] = aoa_scores

temp = esm[esm['Word'].isin(list(a['Word']))]
df_val = a[a['Word'].isin(list(temp['Word']))]
concreteness_scores = [temp[temp['Word'] == w]['norm_concreteness'].values[0] for w in list(df_val['Word'])]
df_val['norm_concreteness'] = concreteness_scores





tokenizer2 = AutoTokenizer.from_pretrained(model1)

tokenizer3 = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")

# Valence_M
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels_valence = df['norm_valence'].values.astype(float)
        self.labels_arousal = df['norm_arousal'].values.astype(float)
        self.labels_dominance = df['norm_dominance'].values.astype(float)
        self.labels_aoa = df['norm_aoa'].values.astype(float)
        self.labels_concreteness = df['norm_concreteness'].values.astype(float)

        self.texts2 = [tokenizer2(str(text).lower(),
                               padding='max_length', max_length = 8, truncation=True,
                                return_tensors="pt") for text in df['Word']]

        self.texts3 = [tokenizer3(str(text).lower(),
                               padding='max_length', max_length = 8, truncation=True,
                                return_tensors="pt") for text in df['Word']]

    def classes(self):
        return self.labels_valence, self.labels_arousal, self.labels_dominance, self.labels_aoa, self.labels_concreteness

    def __len__(self):
        return len(self.labels_valence)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels_valence[idx]), np.array(self.labels_arousal[idx]), np.array(self.labels_dominance[idx]), np.array(self.labels_aoa[idx]), np.array(self.labels_concreteness[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts2[idx], self.texts3[idx]

    def __getitem__(self, idx):

        batch_texts2, batch_texts3 = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)


        return batch_texts2, batch_texts3, batch_y


class BertRegression(nn.Module):

    def __init__(self, dropout=0.1, hidden_dim_valence=1536, hidden_dim_arousal = 1536, hidden_dim_dominance = 1536):

        super(BertRegression, self).__init__()

        self.bert1 = AutoModel.from_pretrained(model1)
        self.bert2 = AutoModel.from_pretrained(model2)
        self.l1 = nn.Linear(hidden_dim_valence, hidden_dim_valence)
        self.l2 = nn.Linear(hidden_dim_valence, hidden_dim_valence)
        self.l3 = nn.Linear(hidden_dim_valence, hidden_dim_valence)

        self.affect = nn.Linear(hidden_dim_valence, 1)
        self.arousal = nn.Linear(hidden_dim_arousal, 1)
        self.dominance = nn.Linear(hidden_dim_dominance, 1)
        self.aoa = nn.Linear(hidden_dim_dominance, 1)
        self.concreteness = nn.Linear(hidden_dim_dominance, 1)
        self.l_1_affect = nn.Linear(hidden_dim_valence, hidden_dim_valence)
        self.l_1_arousal = nn.Linear(hidden_dim_arousal, hidden_dim_arousal)
        self.l_1_dominance = nn.Linear(hidden_dim_dominance, hidden_dim_dominance)
        self.l_1_aoa = nn.Linear(hidden_dim_dominance, hidden_dim_dominance)
        self.l_1_concreteness = nn.Linear(hidden_dim_dominance, hidden_dim_dominance)

        self.layer_norm = nn.LayerNorm(hidden_dim_valence)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id1, mask1, input_id3, mask3):
        _, y = self.bert1(input_ids = input_id1, attention_mask=mask1, return_dict=False)
        _, z = self.bert2(input_ids = input_id3, attention_mask=mask3, return_dict=False)
        x = torch.cat((y, z), dim=1)
        output = from_embedding(x)
        return output
        
       
    def from_embedding(self, x):
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.l1(x) + x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.l2(x) + x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.l3(x) + x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.relu(x)


        affect_all = self.dropout(self.relu(self.layer_norm(self.l_1_affect(x) + x)))
        affect = self.sigmoid(self.affect(affect_all))

        arousal_all = self.dropout(self.relu(self.layer_norm(self.l_1_arousal(x) + x)))
        arousal = self.sigmoid(self.arousal(arousal_all))

        dominance_all = self.dropout(self.relu(self.layer_norm(self.l_1_dominance(x) + x)))
        dominance = self.sigmoid(self.dominance(dominance_all))

        aoa_all = self.dropout(self.relu(self.layer_norm(self.l_1_aoa(x) + x)))
        aoa = self.sigmoid(self.aoa(aoa_all))

        concreteness_all = self.dropout(self.relu(self.layer_norm(self.l_1_concreteness(x) + x)))
        concreteness = self.sigmoid(self.concreteness(concreteness_all))

        return affect, dominance, arousal, aoa, concreteness




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




wandb.init(project="deep_eng", entity="hubertp")
wandb.watch(model, log_freq=5)

h = 0
best_loss = 150
for epoch_num in range(epochs):
    total_loss_train = 0

    for train_input1, train_input2, (valence, arousal, dominance, aoa, conc) in tqdm(train_dataloader):
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
    best_total_corr = 0

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
            total_corr = (total_corr_valence + total_corr_arousal + total_corr_dominance + total_corr_aoa + total_corr_conc) / 5

        # save best models
        if best_total_corr / len(val_dataloader) < total_corr / len(val_dataloader):
            best_total_corr = total_corr
            # delete the previous model
            if os.path.exists('models/deep_eng.pth'):
                os.remove('models/deep_eng.pth')
            torch.save(model.state_dict(), 'models/deep_eng.pth')

    if epoch_num % 2 == 0:
        wandb.log({"loss": total_loss_train / len(df_train), "lr": scheduler.get_last_lr()[0], "epoch": epoch_num, "val_loss": total_loss_val/ len(df_val), "val_corr_valence": total_corr_valence / len(val_dataloader), "val_corr_arousal": total_corr_arousal / len(val_dataloader), "val_corr_dominance": total_corr_dominance / len(val_dataloader), 'val_corr_aoa': total_corr_aoa / len(val_dataloader), 'val_corr_conc': total_corr_conc / len(val_dataloader)})
    print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(df_train): .10f} \
            | Val Loss: {total_loss_val / len(df_val): .10f} | corr_valence: {total_corr_valence / len(val_dataloader): .10f} | corr_arousal: {total_corr_arousal / len(val_dataloader): .10f} | corr_dominance: {total_corr_dominance / len(val_dataloader): .10f} | corr_aoa: {total_corr_aoa / len(val_dataloader): .10f} | corr_conc: {total_corr_conc / len(val_dataloader): .10f}')



model.load_state_dict(torch.load('C:/Users/hplis/PycharmProjects/social_ai/models/deep_eng.pth'))


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    loss = 0
    with torch.no_grad():
        preval, prearo, predom, preaoa, preconc, trueval, truearo, truedom, trueaoa, trueconc = [], [], [], [], [], [], [], [], [], []

        for test_input2, test_input3, (test_valence, test_arousal, test_dominance, test_aoa, test_conc) in test_dataloader:
            mask2 = test_input2['attention_mask'].to(device)
            input_id2 = test_input2['input_ids'].squeeze(1).to(device)
            mask3 = test_input3['attention_mask'].to(device)
            input_id3 = test_input3['input_ids'].squeeze(1).to(device)
            val_label = torch.cat((test_valence, test_arousal, test_dominance, test_aoa, test_conc), dim=0).to(device)

            output1, output2, output3, output4, output5  = model(input_id2, mask2, input_id3, mask3)
            val_output_a = torch.cat((output1, output2, output3, output4, output5), dim=0)
            # batch_loss = criterion(output.float(), val_label.float())

            l1 = criterion1(val_output_a.float(), val_label.view(-1,1).float())
            loss += l1.item()
            preval.extend([p for p in output1.cpu()])
            prearo.extend([p for p in output2.cpu()])
            predom.extend([p for p in output3.cpu()])
            preaoa.extend([p for p in output4.cpu()])
            preconc.extend([p for p in output5.cpu()])
            trueval.extend([t for t in test_valence.cpu()])
            truearo.extend([t for t in test_arousal.cpu()])
            truedom.extend([t for t in test_dominance.cpu()])
            trueaoa.extend([t for t in test_aoa.cpu()])
            trueconc.extend([t for t in test_conc.cpu()])
        print(f'Test Loss: {loss / len(test): .10f}')
            # print loss
    return preval, prearo, predom, preaoa, preconc, trueval, truearo, truedom, trueaoa, trueconc

pred_val, pred_aro, pred_dom, pred_aoa, pred_conc, true_val, true_aro, true_dom, true_aoa, true_conc = evaluate(model, df_test)


diffs = []
for i in range(len(pred_val)):
    diffs.append(float(abs(pred_val[i] - true_val[i])))

mean_val = sum(diffs) / len(diffs)
# compute correlation
p_v = [float(v) for v in pred_val]
t_v = [float(v) for v in true_val]
corr_val = np.corrcoef(p_v, t_v)[0, 1]

diffs = []
for i in range(len(pred_aro)):
    diffs.append(float(abs(pred_aro[i] - true_aro[i])))

p_a = [float(a) for a in pred_aro]
t_a = [float(a) for a in true_aro]
corr_aro = np.corrcoef(p_a, t_a)[0, 1]

mean_aro = sum(diffs) / len(diffs)
# compute correlation

diffs = []
for i in range(len(pred_dom)):
    diffs.append(float(abs(pred_dom[i] - true_dom[i])))

p_d = [float(d) for d in pred_dom]
t_d = [float(d) for d in true_dom]
corr_dom = np.corrcoef(p_d, t_d)[0, 1]

mean_dom = sum(diffs) / len(diffs)


diffs = []
for i in range(len(pred_aoa)):
    diffs.append(float(abs(pred_aoa[i] - true_aoa[i])))

p_ao = [float(ao) for ao in pred_aoa]
t_ao = [float(ao) for ao in true_aoa]
corr_aoa = np.corrcoef(p_ao, t_ao)[0, 1]


diffs = []
for i in range(len(pred_conc)):
    diffs.append(float(abs(pred_conc[i] - true_conc[i])))

p_c = [float(c) for c in pred_conc]
t_c = [float(c) for c in true_conc]
corr_conc = np.corrcoef(p_c, t_c)[0, 1]

mean_conc = sum(diffs) / len(diffs)


print(corr_val)
print(corr_aro)
print(corr_dom)
print(corr_aoa)
print(corr_conc)
