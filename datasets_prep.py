import pandas as pd
import numpy as np

###############################################################################
"""
Prepares the data for training and testing.
"""
###############################################################################
################################################################################
# ENGLISH
#################################

warriner = pd.read_csv('C:/Users/hplis/OneDrive/Desktop/Koła/open_ai/Ratings_Warriner_et_al.csv')
bradley = pd.read_csv('https://raw.githubusercontent.com/mileszim/anew_formats/master/csv/all.csv')

aoa = pd.read_excel('AoA.xlsx')
aoa['norm_aoa'] = (aoa['Rating.Mean'] - min(aoa['Rating.Mean'])) / (max(aoa['Rating.Mean']) - min(aoa['Rating.Mean']))
aoa = aoa.dropna(subset=['norm_aoa'])

esm = pd.read_excel('ESM.xlsx')
esm['norm_concreteness'] =  (esm['Conc.M'] - min(esm['Conc.M'])) / (max(esm['Conc.M']) - min(esm['Conc.M']))
esm = esm.dropna(subset=['norm_concreteness'])

warriner = warriner.dropna(subset=['Word'])

warriner['norm_valence'] = (warriner['V.Mean.Sum'] - min(warriner['V.Mean.Sum'])) / (max(warriner['V.Mean.Sum']) - min(warriner['V.Mean.Sum']))
warriner['norm_arousal'] = (warriner['A.Mean.Sum'] - min(warriner['A.Mean.Sum'])) / (max(warriner['A.Mean.Sum']) - min(warriner['A.Mean.Sum']))
warriner['norm_dominance'] = (warriner['D.Mean.Sum'] - min(warriner['D.Mean.Sum'])) / (max(warriner['D.Mean.Sum']) - min(warriner['D.Mean.Sum']))

np.random.seed(112)

common = [x for x in list(warriner.Word.values) if x in list(bradley.Description.values)]
df_test = warriner[warriner.Word.isin(common)]
warriner = warriner[~warriner.Word.isin(common)]

df_train = warriner.sample(frac=0.9, random_state=42)
df_val = warriner.drop(df_train.index)

temp = aoa[aoa['Word'].isin(list(df_train['Word']))]
a = df_train[df_train['Word'].isin(list(temp['Word']))]
aoa_scores = [temp[temp['Word'] == w]['norm_aoa'].values[0] for w in list(a['Word'])]
a['norm_aoa'] = aoa_scores

temp = esm[esm['Word'].isin(list(a['Word']))]
df_train = a[a['Word'].isin(list(temp['Word']))]
concreteness_scores = [temp[temp['Word'] == w]['norm_concreteness'].values[0] for w in list(df_train['Word'])]
df_train['norm_concreteness'] = concreteness_scores

temp = aoa[aoa['Word'].isin(list(df_test['Word']))]
a = df_test[df_test['Word'].isin(list(temp['Word']))]
aoa_scores = [temp[temp['Word'] == w]['norm_aoa'].values[0] for w in list(a['Word'])]
a['norm_aoa'] = aoa_scores

temp = esm[esm['Word'].isin(list(a['Word']))]
df_test = a[a['Word'].isin(list(temp['Word']))]
concreteness_scores = [temp[temp['Word'] == w]['norm_concreteness'].values[0] for w in list(df_test['Word'])]
df_test['norm_concreteness'] = concreteness_scores

temp = aoa[aoa['Word'].isin(list(df_val['Word']))]
a = df_val[df_val['Word'].isin(list(temp['Word']))]
aoa_scores = [temp[temp['Word'] == w]['norm_aoa'].values[0] for w in list(a['Word'])]
a['norm_aoa'] = aoa_scores

temp = esm[esm['Word'].isin(list(a['Word']))]
df_val = a[a['Word'].isin(list(temp['Word']))]
concreteness_scores = [temp[temp['Word'] == w]['norm_concreteness'].values[0] for w in list(df_val['Word'])]
df_val['norm_concreteness'] = concreteness_scores

df_train['word'] = df_train['Word']
df_test['word'] = df_test['Word']
df_val['word'] = df_val['Word']

# save to parquet
df_train.to_parquet('warriner_anew_train.parquet')
df_test.to_parquet('warriner_anew_test.parquet')
df_val.to_parquet('warriner_anew_val.parquet')

################################################################################
# POLISH
#################################

words_full = pd.read_excel("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4947584/bin/DataSheet1.XLSX", sheet_name="Arkusz1", index_col = 0)
words = words_full.loc[:,[col for col in words_full.columns if not ("Male" in col or "Female" in col or
                                                          "MIN" in col or "MAX" in col or "_N" in col)]]

words = words.rename(columns={"part of speach":"part of speech"}) # Poprawka miss spellingu
print(words.columns) #Jakie mamy informacje
words.head()
words = words.set_index("polish word")
ratings = words.loc[:,[col for col in words.columns if "M" in col or "part of speech" in col or 'SD' in col]]# Wybieranie samych średnich ocen

ratings = ratings.drop("part of speech", axis=1)
ratings = (ratings-ratings.mean())/ratings.std() #normalize ratings
ratings_nouns = ratings.loc[words["part of speech"]=="N",:] #Same rzeczowniki

ratings.reset_index(level=0, inplace=True)

ratings['norm_valence'] = (ratings['Valence_M'] - min(ratings['Valence_M'])) / (max(ratings['Valence_M']) - min(ratings['Valence_M']))
ratings['norm_arousal'] = (ratings['arousal_M'] - min(ratings['arousal_M'])) / (max(ratings['arousal_M']) - min(ratings['arousal_M']))
ratings['norm_dominance'] = (ratings['dominance_M'] - min(ratings['dominance_M'])) / (max(ratings['dominance_M']) - min(ratings['dominance_M']))
ratings['norm_origin'] = (ratings['origin_M'] - min(ratings['origin_M'])) / (max(ratings['origin_M']) - min(ratings['origin_M']))
ratings['norm_significance'] = (ratings['significance_M'] - min(ratings['significance_M'])) / (max(ratings['significance_M']) - min(ratings['significance_M']))
ratings['norm_concreteness'] = (ratings['concretness_M'] - min(ratings['concretness_M'])) / (max(ratings['concretness_M']) - min(ratings['concretness_M']))
ratings['norm_imageability'] = (ratings['imegability_M'] - min(ratings['imegability_M'])) / (max(ratings['imegability_M']) - min(ratings['imegability_M']))
ratings['norm_aqcuisition'] = (ratings['ageOfAquisition_M'] - min(ratings['ageOfAquisition_M'])) / (max(ratings['ageOfAquisition_M']) - min(ratings['ageOfAquisition_M']))

ratings['word'] = ratings['polish word']

np.random.seed(112)
df_train, df_val, df_test = np.split(ratings.sample(frac=1, random_state=42),
                                     [int(.8*len(ratings)), int(.9*len(ratings))])

# save
df_train.to_parquet('train_octa_clean.parquet')
df_val.to_parquet('val_octa_clean.parquet')
df_test.to_parquet('test_octa_clean.parquet')

################################################################################
# SPANISH
#################################

df = pd.read_excel('data/Redondo-BRM-2007/concaro.xlsx')

df['norm_valence'] = (df['VAL_M'] - min(df['VAL_M'])) / (max(df['VAL_M']) - min(df['VAL_M']))
df['norm_arousal'] = (df['ARO_M'] - min(df['ARO_M'])) / (max(df['ARO_M']) - min(df['ARO_M']))
df['norm_concreteness'] = (df['CON_M'] - min(df['CON_M'])) / (max(df['CON_M']) - min(df['CON_M']))
df['norm_imagebility'] = (df['IMA_M'] - min(df['IMA_M'])) / (max(df['IMA_M']) - min(df['IMA_M']))
df['norm_familiarity'] = (df['FAM_M'] - min(df['FAM_M'])) / (max(df['FAM_M']) - min(df['FAM_M']))
df['word'] = df['Word']

np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8*len(df)), int(.9*len(df))])
# save
df_train.to_parquet('train_spanish.parquet')
df_val.to_parquet('val_spanish.parquet')
df_test.to_parquet('test_spanish.parquet')

################################################################################
# DUTCH
#################################

data = pd.read_excel('data/13428_2012_243_MOESM1_ESM.xlsx')
concrete = pd.read_excel('data/dutch_concrete.xlsx')

data.columns = data.iloc[0]
data = data.iloc[1:]
data = data.iloc[:, : 13]
data = data.dropna()

data['norm_valence'] = (data['M V'] - min(data['M V'])) / (max(data['M V']) - min(data['M V']))
data['norm_arousal'] = (data['M A'] - min(data['M A'])) / (max(data['M A']) - min(data['M A']))
data['norm_dominance'] = (data['M P'] - min(data['M P'])) / (max(data['M P']) - min(data['M P']))
data['norm_age_of_aquisition'] = (data['M AoA'] - min(data['M AoA'])) / (max(data['M AoA']) - min(data['M AoA']))

np.random.seed(112)
df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42),
                                     [int(.8*len(data)), int(.9*len(data))])

# save
df_train.to_parquet('train_dutch.parquet')
df_val.to_parquet('val_dutch.parquet')
df_test.to_parquet('test_dutch.parquet')