from repo.utils import evaluate, get_metrics
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, PreTrainedTokenizerFast, RobertaTokenizer, RobertaModel
from repo.dataset_and_model import BertRegression
import torch
import pandas as pd
from transformers import logging
import os
logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

###############################################################################
"""
Calculates the results for all languages
"""
###############################################################################
###############################################################################

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
df_test = pd.read_parquet('data/test_spanish.parquet')

# INITIALIZATION
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/spanish.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/spanish_results.csv', index=False)

###############################################################################
# POLISH
#################################
print("Polish")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'origin', 'significance', 'concreteness', 'imageability', 'aqcuisition']
max_len = 10
hidden_dim = 768
dropout = 0.2
model_dir = "D:/PycharmProjects/roberta/roberta_base_transformers/"
model_name = ["bert"]
model_initialization = [RobertaModel.from_pretrained('D:/PycharmProjects/roberta/roberta_base_transformers/')]

# DATA LOADING
df_test = pd.read_parquet('data/test_octa_clean.parquet')

# INITIALIZATION
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
tokenizer.pad_token = 0
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/octa_clean.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/polish_results.csv', index=False)



###############################################################################
# ENGLISH STIMULI DESCENT
#################################
print("English Stimuli Descent")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']
max_len = 8
hidden_dim = 768
dropout = 0.1
model_dir = "finiteautomata/bertweet-base-emotion-analysis"
model_name = ["bert"]
model_initialization = [AutoModel.from_pretrained("finiteautomata/bertweet-base-emotion-analysis")]

# DATA LOADING
df_test = pd.read_parquet('data/warriner_anew_test.parquet')

# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/english_stimuli_descent.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/stimuli_descent_results.csv', index=False)


###############################################################################
# ENGLISH nghuyong
#################################
print("English Stimuli Descent")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']
max_len = 8
hidden_dim = 768
dropout = 0.1
model_dir = "nghuyong/ernie-2.0-base-en"    # "nghuyong/ernie-2.0-en"
model_name = ["bert"]
model_initialization = [AutoModel.from_pretrained("nghuyong/ernie-2.0-base-en")]

# DATA LOADING
df_test = pd.read_parquet('data/warriner_anew_test.parquet')

# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/english_one_nghuyong'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/english_results.csv', index=False)


#################################
print("Dutch 1")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aqcuisition']
max_len = 8
hidden_dim = 768
dropout = 0.1
model_dir = "GroNLP/bert-base-dutch-cased"
model_name = ['bert2']
model_initialization = [AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")]


# DATA LOADING
df_test = pd.read_parquet('data/test_dutch.parquet')

# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
save_dir = 'models/gronlp_ducth_base.pth'

model.load_state_dict(torch.load(save_dir))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/dutch_results.csv', index=False)

###############################################################################
print("German")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'imageability']
max_len = 7
hidden_dim = 768
dropout = 0.1
model_dir = "dbmdz/bert-base-german-uncased"

model_name = ['bert']
model_initialization = [AutoModel.from_pretrained("dbmdz/bert-base-german-uncased")]

# DATA LOADING
df_test = pd.read_parquet('data/test_german.parquet')

# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
save_dir = 'models/german'

model.load_state_dict(torch.load(save_dir))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/german_results.csv', index=False)

#################################
print("French")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal']
max_len = 9
hidden_dim = 768
dropout = 0.1
model_dir = "EIStakovskii/french_toxicity_classifier_plus_v2"

model_name = ['bert']

model_initialization = [AutoModel.from_pretrained("EIStakovskii/french_toxicity_classifier_plus_v2")]

# DATA LOADING
df_test = pd.read_parquet('data/test_french.parquet')

# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
save_dir = 'models/french'

model.load_state_dict(torch.load(save_dir))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/french_results.csv', index=False)



##################################
######## ABSTRACTNESS TESTS
##################################

##### ABSTRACT CHECK
print("Abstractness check")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aoa']
max_len = 8
hidden_dim = 768
dropout = 0.1
model_dir = "finiteautomata/bertweet-base-emotion-analysis"
model_name = ["bert"]
model_initialization = [AutoModel.from_pretrained("finiteautomata/bertweet-base-emotion-analysis")]

# DATA LOADING
df_test = pd.read_parquet('data/abstractness_robustness_test_test.parquet')

# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/abstractness_robustness_check2.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE RESULTS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/abstractness_check_results.csv', index=False)




##### ABSTRACT COMPARE
print("Abstractness compare")

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aoa']
max_len = 8
hidden_dim = 768
dropout = 0.1
model_dir = "finiteautomata/bertweet-base-emotion-analysis"
model_name = ["bert"]
model_initialization = [AutoModel.from_pretrained("finiteautomata/bertweet-base-emotion-analysis")]

# DATA LOADING
df_test = pd.read_parquet('data/abstractness_robustness_compare_test.parquet')

# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/abstractness_robustness_compare.pth'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)

# CALCULATING CORRELATIONS
get_metrics(predictions, labels, metric_names)

# SAVE PREDICTIONS
predictions = [[float(p2) for p2 in p] for p in predictions]
labels = [[float(l2) for l2 in l] for l in labels]
# save predictions and labels
columns = metric_names
columns.extend([metric + '_label' for metric in metric_names])
column_values = predictions
column_values.extend(labels)
df = pd.DataFrame()
for column, column_value in zip(columns, column_values):
    df[column] = column_value

# save
df.to_csv('predictions_results/abstractness_compare_results.csv', index=False)

##################################
######## QUESTIONNAIRE TEST
##################################
# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'origin', 'significance', 'concreteness', 'imageability', 'aqcuisition']
max_len = 10
hidden_dim = 768
dropout = 0.2
model_dir = "D:/PycharmProjects/roberta/roberta_base_transformers/"
model_name = ["bert"]
model_initialization = [RobertaModel.from_pretrained('D:/PycharmProjects/roberta/roberta_base_transformers/')]

# DATA LOADING
df_test = pd.read_excel('data/questionnaire_words.xlsx')
df_test.columns = ['word']
# INITIALIZATION
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
tokenizer.pad_token = 0
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/octa_clean.pth'))

for metric in metric_names:
    df_test['norm_' + metric] = 0

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)
import pandas as pd

predictions = [[float(p2) for p2 in p] for p in predictions]

columns = metric_names
column_values = predictions
for column, column_value in zip(columns, column_values):
    df_test[column] = [c[0] for c in column_value]

columns.append('word')
df_test = df_test[columns]

# save
df_test.to_csv('predictions_results/questionnaire_results.csv', index=False)


############################################################################################################
######################################## ENGLISH AOA TEST ##################################################
############################################################################################################

# HYPERPARAMETERS
metric_names = ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']
max_len = 8
hidden_dim = 768
dropout = 0.1
model_dir = "nghuyong/ernie-2.0-base-en"    # "nghuyong/ernie-2.0-en"
model_name = ["bert"]
model_initialization = [AutoModel.from_pretrained("nghuyong/ernie-2.0-base-en")]

# DATA LOADING
df_test = pd.read_excel('data/AOA_Kuperman.xlsx')
df_test['word'] = df_test['Word']
del df_test['Word']

columns = list(df_test.columns)

temporary_columns = []
for metric in metric_names:
    temporary_columns.append('norm_' + metric)
    df_test['norm_' + metric] = 0


# INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertRegression(model_name, model_initialization, metric_names, dropout, hidden_dim)
model.load_state_dict(torch.load('models/english_one_nghuyong'))

# EVALUATION
predictions, labels = evaluate(model, tokenizer, df_test, max_len, metric_names)


predictions = [[float(p2) for p2 in p] for p in predictions]

column_values = predictions
for column, column_value in zip(metric_names, column_values):
    columns.append(column)
    df_test[column] = [c for c in column_value]

df_test = df_test[columns]

# save
df_test.to_csv('predictions_results/english_aoa_results.csv', index=False)