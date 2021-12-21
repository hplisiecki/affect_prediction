import pandas as pd
import numpy as np
import wandb
# print(np.abs(np.random.randn(1000000)).mean())
# load csv from internet
warriner = pd.read_csv('C:/Users/hplis/OneDrive/Desktop/Koła/open_ai/Ratings_Warriner_et_al.csv')





# reformat to 0 to 1
warriner['norm_valence'] = warriner['V.Mean.Sum'] / 9
warriner['norm_arousal'] = warriner['A.Mean.Sum'] / 9
warriner['norm_dominance'] = warriner['D.Mean.Sum'] / 9

warriner['norm_dominance_sd'] = (warriner['D.SD.Sum'] - min(warriner['D.SD.Sum'])) / (max(warriner['D.SD.Sum']) - min(warriner['D.SD.Sum']))
warriner['norm_arousal_sd'] = (warriner['A.SD.Sum'] - min(warriner['A.SD.Sum'])) / (max(warriner['A.SD.Sum']) - min(warriner['A.SD.Sum']))
warriner['norm_valence_sd'] = (warriner['V.SD.Sum'] - min(warriner['V.SD.Sum'])) / (max(warriner['V.SD.Sum']) - min(warriner['V.SD.Sum']))

# print("MAE przewiwydania średniej Valence dla ANEW: ",(warriner['Valence Mean']-warriner['Valence Mean'].mean()).abs().mean())

import torch
import numpy as np
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Valence_M
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels_valence = df['norm_valence'].values.astype(float)
        self.labels_arousal = df['norm_arousal'].values.astype(float)
        self.labels_dominance = df['norm_dominance'].values.astype(float)

        self.labels_dominance_sd = df['norm_dominance_sd'].values.astype(float)
        self.labels_valence_sd = df['norm_valence_sd'].values.astype(float)
        self.labels_arousal_sd = df['norm_arousal_sd'].values.astype(float)

        self.texts = [tokenizer(str(text),
                               padding='max_length', max_length = 9, truncation=True,
                                return_tensors="pt") for text in df['Word']]

    def classes(self):
        return self.labels_valence, self.labels_arousal, self.labels_dominance, self.labels_dominance_sd, self.labels_valence_sd, self.labels_arousal_sd

    def __len__(self):
        return len(self.labels_valence)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels_valence[idx]), np.array(self.labels_arousal[idx]), np.array(self.labels_dominance[idx]), np.array(self.labels_dominance_sd[idx]), np.array(self.labels_valence_sd[idx]), np.array(self.labels_arousal_sd[idx])

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
df_train, df_val, df_test = np.split(warriner.sample(frac=1, random_state=42),
                                     [int(.8*len(warriner)), int(.9*len(warriner))])

# save
# df_train.to_csv('train_warriner.csv', index=False)
# df_val.to_csv('val_warriner.csv', index=False)
# df_test.to_csv('test_warriner.csv', index=False)

df_train = pd.read_csv('train_warriner.csv')
df_val = pd.read_csv('val_warriner.csv')
df_test = pd.read_csv('test_warriner.csv')


class BertRegression(nn.Module):

    def __init__(self, dropout=0.2, hidden_dim=768):

        super(BertRegression, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.l1 = nn.Linear(768, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.affect = nn.Linear(hidden_dim, 1)
        self.arousal = nn.Linear(hidden_dim, 1)
        self.dominance = nn.Linear(hidden_dim, 1)
        self.af_sd = nn.Linear(hidden_dim, 1)
        self.ar_sd = nn.Linear(hidden_dim, 1)
        self.do_sd = nn.Linear(hidden_dim, 1)
        self.l_1_affect = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_arousal = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_dominance = nn.Linear(hidden_dim, hidden_dim)
        self.l_2_affect = nn.Linear(hidden_dim, hidden_dim)
        self.l_2_arousal = nn.Linear(hidden_dim, hidden_dim)
        self.l_2_dominance = nn.Linear(hidden_dim, hidden_dim)



        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, x = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
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

        affect_all = self.relu(self.dropout(self.layer_norm(self.l_1_affect(x) + x)))
        arousal_all = self.relu(self.dropout(self.layer_norm(self.l_1_arousal(x) + x)))
        dominance_all = self.relu(self.dropout(self.layer_norm(self.l_1_dominance(x) + x)))

        # affect_all = self.relu(self.dropout(self.layer_norm(self.l_2_affect(affect_all) + affect_all)))
        # arousal_all = self.relu(self.dropout(self.layer_norm(self.l_2_arousal(arousal_all) + arousal_all)))
        # dominance_all = self.relu(self.dropout(self.layer_norm(self.l_2_dominance(dominance_all) + dominance_all)))

        affect = self.sigmoid(self.affect(affect_all))
        arousal = self.sigmoid(self.arousal(arousal_all))
        dominance = self.sigmoid(self.dominance(dominance_all))


        affect_sd = self.sigmoid(self.af_sd(affect_all))
        arousal_sd = self.sigmoid(self.ar_sd(arousal_all))
        dominance_sd = self.sigmoid(self.do_sd(dominance_all))



        return affect, arousal, dominance, affect_sd, arousal_sd, dominance_sd



from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

epochs = 500
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

wandb.init(project="affect_anew", entity="hubertp")
wandb.watch(model, log_freq=5)

# for param in model.bert.parameters():
#     print(param.requires_grad)
#     param.requires_grad = False

h = 1
for epoch_num in range(epochs):
    total_loss_train = 0

    for train_input, (valence, arousal, dominance, val_sd, ar_sd, dom_sd) in tqdm(train_dataloader):
        if epoch_num == 48:
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=1e-5,
                                          eps=1e-8,
                                          weight_decay=0.3,
                                          amsgrad=True)
            for param in model.bert.parameters():
                param.requires_grad = True
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=600,
                                                        num_training_steps=len(train_dataloader) * epochs)
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)
        train_label = torch.cat((valence, arousal, dominance), dim=0).to(device)
        sd_label = torch.cat((val_sd, ar_sd, dom_sd), dim=0).to(device)
        output1, output2, output3, output4, output5, output6 = model(input_id, mask)
        # concatenate
        output_a = torch.cat((output1, output2, output3), dim=0)
        l1 = criterion(output_a.float(), train_label.view(-1,1).float())
        output_b = torch.cat((output4, output5, output6), dim=0)
        l2 = criterion(output_b.float(), sd_label.view(-1,1).float())
        batch_loss = l1 + h * l2
        total_loss_train += batch_loss.item()

        # try after 5 epochs
        if h > 0:
            # if epoch_num > 50:
            h = h - 0.005


        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
    total_acc_val = 0
    total_loss_val = 0

    with torch.no_grad():

        for val_input, (val_valence, val_arousal, val_dominance, val_ar_sd, val_ar_sd, val_dom_sd) in val_dataloader:
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            val_label = torch.cat((val_valence, val_arousal, val_dominance), dim=0).to(device)
            sd_label = torch.cat((val_ar_sd, val_ar_sd, val_dom_sd), dim=0).to(device)

            output1, output2, output3, output4, output5, output6 = model(input_id, mask)
            val_output_a = torch.cat((output1, output2, output3), dim=0)
            l1 = criterion(val_output_a.float(), val_label.view(-1,1).float())
            val_output_b = torch.cat((output4, output5, output6), dim=0)
            l2 = criterion(val_output_b.float(), sd_label.view(-1,1).float())

            batch_loss = l1 + h * l2
            total_loss_val += batch_loss.item()

    if epoch_num % 2 == 0:
        wandb.log({"loss": total_loss_train / len(df_train), "lr": scheduler.get_last_lr()[0], "epoch": epoch_num, "val_loss": total_loss_val/ len(df_val)})
    print(
        f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(df_train): .10f} \
            | Val Loss: {total_loss_val / len(df_val): .10f}')




def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        preval, prearo, predom, trueval, truearo, truedom = [], [], [], [], [], []

        for test_input, (test_valence, test_arousal, test_dominance, v_, a_, d_) in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output1, output2, output3, output4, output5, output6 = model(input_id, mask)
            # batch_loss = criterion(output.float(), val_label.float())


            preval.extend([p for p in output1.cpu()])
            prearo.extend([p for p in output2.cpu()])
            predom.extend([p for p in output3.cpu()])
            trueval.extend([t for t in test_valence.cpu()])
            truearo.extend([t for t in test_arousal.cpu()])
            truedom.extend([t for t in test_dominance.cpu()])

            # print loss
    return preval, prearo, predom, trueval, truearo, truedom

pred_val, pred_aro, pred_dom, true_val, true_aro, true_dom = evaluate(model, df_test)

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


print(corr_val)
print(corr_aro)
print(corr_dom)

bradley = pd.read_csv('https://raw.githubusercontent.com/mileszim/anew_formats/master/csv/all.csv')

# reformat to 0 to 1
bradley['norm_valence'] = bradley['Valence Mean'] / 9
bradley['norm_arousal'] = bradley['Arousal Mean'] / 9
bradley['norm_dominance'] = bradley['Dominance Mean'] / 9

bradley['norm_dominance_sd'] = (bradley['Dominance SD'] - min(bradley['Dominance SD'])) / (max(bradley['Dominance SD']) - min(bradley['Dominance SD']))
bradley['norm_arousal_sd'] = (bradley['Arousal SD'] - min(bradley['Arousal SD'])) / (max(bradley['Arousal SD']) - min(bradley['Arousal SD']))
bradley['norm_valence_sd'] = (bradley['Valence SD'] - min(bradley['Valence SD'])) / (max(bradley['Valence SD']) - min(bradley['Valence SD']))
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels_valence = df['norm_valence'].values.astype(float)
        self.labels_arousal = df['norm_arousal'].values.astype(float)
        self.labels_dominance = df['norm_dominance'].values.astype(float)

        self.labels_dominance_sd = df['norm_dominance_sd'].values.astype(float)
        self.labels_valence_sd = df['norm_valence_sd'].values.astype(float)
        self.labels_arousal_sd = df['norm_arousal_sd'].values.astype(float)

        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 9, truncation=True,
                                return_tensors="pt") for text in df['Description']]

    def classes(self):
        return self.labels_valence, self.labels_arousal, self.labels_dominance, self.labels_dominance_sd, self.labels_valence_sd, self.labels_arousal_sd

    def __len__(self):
        return len(self.labels_valence)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels_valence[idx]), np.array(self.labels_arousal[idx]), np.array(self.labels_dominance[idx]), np.array(self.labels_dominance_sd[idx]), np.array(self.labels_valence_sd[idx]), np.array(self.labels_arousal_sd[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y







def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        preval, prearo, predom, trueval, truearo, truedom = [], [], [], [], [], []

        for test_input, (test_valence, test_arousal, test_dominance, v_, a_, d_) in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output1, output2, output3, output4, output5, output6 = model(input_id, mask)
            # batch_loss = criterion(output.float(), val_label.float())


            preval.extend([p for p in output1.cpu()])
            prearo.extend([p for p in output2.cpu()])
            predom.extend([p for p in output3.cpu()])
            trueval.extend([t for t in test_valence.cpu()])
            truearo.extend([t for t in test_arousal.cpu()])
            truedom.extend([t for t in test_dominance.cpu()])

            # print loss
    return preval, prearo, predom, trueval, truearo, truedom


pred_val, pred_aro, pred_dom, true_val, true_aro, true_dom = evaluate(model, bradley)

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


print(corr_val)
print(corr_aro)
print(corr_dom)
