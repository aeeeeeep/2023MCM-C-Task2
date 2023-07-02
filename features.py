import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import holidays
import datetime

df = pd.read_excel('./data/Problem_C_Data_Wordle.xlsx')
unigram_freq = pd.read_csv('./data/unigram_freq.csv')
df['Date'] = df['Date'].apply(pd.to_datetime)

y1 = df.loc[:,'1 try']
y2 = df.loc[:,'2 tries']
y3 = df.loc[:,'3 tries']
y4 = df.loc[:,'4 tries']
y5 = df.loc[:,'5 tries']
y6 = df.loc[:,'6 tries']
y7 = df.loc[:,'7 or more tries (X)']
x = np.array(range(0, len(y1), 1))
labels = ['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)']

df = pd.merge(df, unigram_freq, how='left')

# '''
# 词性分类
import hanlp
tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
Word_Cls = pd.Series(tagger(df['Word'].tolist()), dtype="category")
Word_Cls = Word_Cls.cat.rename_categories([4,1,5,14,6,9,15,2,3,12,11,10,7,8,13])
df['WordCls'] = Word_Cls
# '''

# 音节统计
import textstat
ns = []
for word in df['Word']:
    ns.append(textstat.syllable_count(word))
df['Syllable'] = ns

# 情感分析
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('sentiwordnet') 
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet

def get_sentiment_score(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    synset = synsets[0]
    swn_synset = sentiwordnet.senti_synset(synset.name())
    return (swn_synset.pos_score(), swn_synset.neg_score())

def classify_word(word):
    scores = get_sentiment_score(word)
    if scores is None:
        if word == 'hunky':
            return 1
        else:
            return 0
    pos_score, neg_score = scores
    if pos_score > neg_score:
        return 1
    elif neg_score > pos_score:
        return -1
    else:
        return 0

se = []
for word in df['Word']:
    se.append(classify_word(word))
df['Sentiment'] = se

repeated = []
for word in df['Word']:
    repeated.append(sum(1 for letter in set(word) if word.count(letter) > 1))
df['Repeated'] = repeated

# 假期
holiday_list = []
for holiday in holidays.US(years=[2022]).items():
    holiday_list.append(holiday)

holiday_df = pd.DataFrame(holiday_list, columns=["Date", "Holiday"])
holiday_df['Date'] = pd.to_datetime(holiday_df['Date'])

start_date = datetime.date(2022, 1, 7)
end_date = datetime.date(2022, 12, 31)
date_range = pd.date_range(start_date, end_date)

weekend_df = pd.DataFrame(index=date_range,columns=['Holiday'])
weekend_df['Date'] = date_range
for date in date_range:
    datetime_obj = date.to_pydatetime()
    day_of_week = datetime_obj.weekday()
    if day_of_week == 5 or day_of_week == 6:
        weekend_df.loc[date, 'Holiday'] = 1
    else:
        weekend_df.loc[date, 'Holiday'] = 0

for date in holiday_df['Date']:
    weekend_df.loc[date,'Holiday'] = 1

df = pd.merge(df, weekend_df, how='left')

# 正态拟合
from scipy.optimize import curve_fit

tries = df.iloc[:, 5:12]

def normal_distribution(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

popts0, popts1 = [], []
for i in range(len(tries)):
    t = np.array(tries.iloc[i,:], dtype='float32')
    t_norm = t/100
    x = np.linspace(0,len(t_norm),100)
    popt, pcov = curve_fit(normal_distribution, np.arange(len(t_norm)), t_norm)
    fit = normal_distribution(x, *popt)
    popts0.append(popt[0])
    popts1.append(popt[1])
df['popts0'] = popts0
df['popts1'] = popts1
# print(popts0, popts1)

df.to_excel("./data/df.xlsx", index=False)
