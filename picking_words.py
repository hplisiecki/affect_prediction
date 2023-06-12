import pandas as pd

# read from text

df = pd.read_csv('study_2_data\Kazojc2009.txt', sep='\t', header=None)
words = [line.split('=')[0] for line in df[0]]
values = [line.split('=')[1] for line in df[0]]
df = pd.DataFrame({'word': words, 'value': values})
df['value'] = df['value'].astype(float)

df = df[df['value'] > 5]
# sort by value
df = df.sort_values(by=['value'], ascending=False)
# read from text
with open('D:\PycharmProjects\social_ai\data\stopwords.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

stopwords = [lines.replace('\n', '') for lines in lines]

# remove stopwords
df = df[~df['word'].isin(stopwords)]


sample_words = df['word'].sample(200)
sample_words = pd.DataFrame({'word': sample_words.values})

words_full = pd.read_excel("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4947584/bin/DataSheet1.XLSX", sheet_name="Arkusz1", index_col = 0)
words = words_full.loc[:,[col for col in words_full.columns if not ("Male" in col or "Female" in col or
                                                          "MIN" in col or "MAX" in col or "_N" in col)]]
words['polish word'] = words['polish word'].str.lower()

sample_words['Imbir'] = [True if word in words['polish word'].values else False for word in sample_words['word']]

# save
sample_words.to_csv('D:\PycharmProjects\social_ai\data\sample_words.csv', index=False)
