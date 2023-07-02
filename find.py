import holidays
import pandas as pd
import hanlp
import textstat
import datetime
# first run:
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('sentiwordnet')
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet

def find(word, date=0, need_date=True):
    attribute = []
    # 词频统计
    unigram_freq = pd.read_csv('./data/unigram_freq.csv')
    try:
        attribute.append(int(unigram_freq.loc[unigram_freq["Word"] == word]["Count"]))
    except:
        attribute.append(int(12000))

    # '''
    # 词性分类
    tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
    tagger_dict = {'DT': 4, 'EX': 1, 'IN': 5, 'JJ': 14, 'JJR': 6, 'MD': 9, 'NN': 15, 'NMS': 2, 'PRP$': 3, 'RB': 12,
                   'VB': 11, 'VBD': 10, 'VBG': 7, 'VBN': 8, 'VBP': 13}
    try:
        attribute.append(int(tagger_dict[tagger([word])[0]]))
    except:
        attribute.append(int(1))
    # '''

    # 音节统计
    attribute.append(int(textstat.syllable_count(word)))

    # 情感分析
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

    attribute.append(int(classify_word(word)))

    # 重复字母
    attribute.append(int(sum(1 for letter in set(word) if word.count(letter) > 1)))

    # 假期
    if need_date:
        is_holiday = 0
        format = "%Y-%m-%d"
        date = datetime.datetime.strptime(date, format)
        if date.weekday() in [5, 6]:
            is_holiday = 1
        us_holiday = holidays.US()
        if us_holiday.get(date) != None:
            is_holiday = 1
        attribute.append(int(is_holiday))
    else:
        attribute.append(int(0))

    return attribute


if __name__ == "__main__":
    df = pd.read_json('./data/words.json')
    finds = []
    for i in df['words']:
        finds.append(find("eerie", date=0, need_date=False))
    print(finds.shape)
    # df[]
    # df.to_excel("./data/finds.xlsx", index=False)
