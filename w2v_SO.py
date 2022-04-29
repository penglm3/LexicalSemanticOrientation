import pandas as pd
import re
import jieba
import pickle as pkl
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np
import os
import time
import glob

"""
PENG Limin, 4th March, 2022
"""
"""
os.chdir('D:\\PLM\\宏观金融seminar\\Seminar2021-2022\\央行沟通词典\\code')

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

# load user-defined word_list
stopwords = [line.strip() for line in open(r'dict\百度停用词表中文.txt', 'r', encoding='utf-8-sig').readlines()]  # 停用词词典
Pwords = [line.strip() for line in open(r'dict\LM_pos.txt', 'r', encoding='utf-8-sig').readlines()]
Nwords = [line.strip() for line in open(r'dict\LM_neg.txt', 'r', encoding='utf-8-sig').readlines()]
Pwords2 = [line.strip() for line in open(r'dict\INF_pos.txt', 'r', encoding='utf-8-sig').readlines()]
Nwords2 = [line.strip() for line in open(r'dict\INF_neg.txt', 'r', encoding='utf-8-sig').readlines()]
degreeDict = [line.strip() for line in open(r'dict\degreeDict.txt', 'r', encoding='utf-8-sig').readlines()]
notDict = [line.strip() for line in open(r'dict\notDict.txt', 'r', encoding='utf-8-sig').readlines()]
stopwords = list(set(stopwords)-set(Pwords)-set(Nwords)-set(degreeDict)-set(degreeDict)-set(notDict)-set(Pwords2)-set(Nwords2)) # 去重后的停用词
"""

def tidy_data():
    zhengzhi = pd.read_csv("data\\中央政治局会议文本.csv",parse_dates=['date'])
    zhengzhi['date'] = zhengzhi['date'].apply(lambda x:x.replace("年","/").replace("月","/").replace("日",""))
    zhengzhi['date'] = pd.to_datetime(zhengzhi['date'])
    zhengzhi['type'] = "政治局会议"

    jingji = pd.read_csv("data\\中央经济工作会议文本.csv",parse_dates=[['year','end_date']])
    jingji = jingji.rename(columns={'year_end_date':'date'}) # year_end_date is year & end_date
    del jingji["begin_date"]
    jingji['type'] = "经济工作会议"

    changwu = pd.read_csv("data\\国务院常务会议文本.csv",parse_dates=['meeting_date','report_date'])
    changwu = changwu.rename(columns={'meeting_date':'date'})
    for col in ["title","report_date", "name"]:
        del changwu[col]
    changwu['type'] = "国务院常务会议"

    jingji.sort_values(by='date',inplace=True)
    jingji.reset_index(drop=True,inplace=True)
    zhengzhi.sort_values(by='date',inplace=True)
    zhengzhi.reset_index(drop=True,inplace=True)
    changwu.sort_values(by='date',inplace=True)
    changwu.reset_index(drop=True,inplace=True)

    corpus = jingji.append(zhengzhi,ignore_index=True).append(changwu,ignore_index=True)
    corpus.reset_index(drop=True,inplace=True)
    pkl.dump(corpus,open("data\\会议文本.pkl","wb"))

def zscore(arr):
    """
    normalized values of the given array
    :param arr: array
    :return: normalized values of the given array
    """
    mu = arr.mean()
    sigma = arr.std()
    return (arr - mu) / sigma

def clean_text_wv(text, n_gram=0):
    """
    把文本划分成句子单位，在句子中分词（word2vec输入格式）
    :param text: a document
    :param ngram: if specified, number-gram, int
    :return: a list of lists, words of each sentence is stored in a list
    """
    text_ = ''.join(''.join(text.split('\n')).split('\x0c')) #去掉换行符换页符
    #remove_digits = str.maketrans('', '', digits)
    #text = text.translate(remove_digits) #去掉数字(页码影响分词)
    text_ = text_.replace(" ", "") #去掉空格
    #sentences = re.findall(zhon.hanzi.sentence, text) # list of lists
    sentences = re.split(r'(\!|\?|。|！|？)', text_) # 断句
    doc_list = []
    for sentence in sentences:
        res = re.compile("[^\u4e00-\u9fa5^a-z^A-Z]")  # 去掉其他,仅保留中英文
        sentence = res.sub("", sentence)
        if n_gram:
            word_list = gen_ngram(sentence, n_gram)
        else:
            word_list = jieba.lcut(sentence,cut_all=False,HMM=True)
        word_list = [w for w in word_list if w not in stopwords and len(w)>1]
        if len(word_list)>1:
            doc_list.append(word_list)
    return doc_list

def gen_trigram(text, min_count,threshold):
    """
    form phrases automatically, using gensim's `Phrases`
    :param text: list of lists
    :param min_count: 保留单词的最低频次
    :param threshold: 组合短语的最低阈值
    :return: list of lists
    """
    # Build the bigram and trigram models
    bigram = Phrases(text, min_count=min_count, threshold=threshold)  # higher threshold fewer phrases.
    trigram = Phrases(bigram[text], threshold=threshold)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    return(list(trigram_mod[bigram_mod[text]]))

def compare_dict(dict_path1,dict_path2):
    """
    pairwise comparison, calculates numbers of words in common
    :param dict_path1: path of the 1st dict (.txt), string
    :param dict_path2: path of the 2nd dict (.txt), string
    :return:
    """
    word_list1 = [line.strip() for line in open(dict_path1, 'r', encoding='utf-8').readlines()]
    word_list2 = [line.strip() for line in open(dict_path2, 'r', encoding='utf-8').readlines()]
    word1 = set(word_list1)
    word2 = set(word_list2)
    len1 = len(word_list1)
    len2 = len(word_list2)
    word1_unique = word1-word2
    num_word1_unique = len(set(word1_unique)) # counts of word_list1's unique words
    word2_unique = word2 - word1
    num_word2_unique = len(set(word2_unique))  # counts of word_list2's unique words
    word_common = word1 & word2
    num_word_common = len(set(word_common))  # counts of common words
    print("comparing {} and {} ".format(dict_path1,dict_path2))
    print("num of words in word_list 1: {}".format(len1))
    print("num of words in word_list 2: {}".format(len2))
    print("num of unique words in word_list 1: {}".format(num_word1_unique))
    print("num of unique words in word_list 2: {}".format(num_word2_unique))
    print("num of common words: {}".format(num_word_common))

def build_dict(word_list,filename):
    """
    default, newly built dicts are saved under the folder "dict\\new"
    """
    print("building dict{}".format(filename+'.txt'))
    print("dict size:{} words".format(len(word_list)))
    with open("dict\\new\\{}.txt".format(filename), "a",encoding='utf-8-sig') as f:
        for line in word_list:
            f.write(str(line) + '\n')
    f.close()

def gen_random_dict(model,num_pos,num_neg,pos_dict_name,neg_dict_name):
    """
    draw words randomly, to build (random) lexicons
    :param model: trained model, MODEL file
    :param num_pos: dict size of the pseudo_pos_list, int
    :param num_neg: dict size of the pseudo_neg_list, int
    :param pos_dict_name: filename of the pseudo positive dict, string
    :param neg_dict_name: filename of the pseudo negative dict, string
    :return: pseudo_pos_list, pseudo_neg_list
    """
    import random
    vocab = list(model.wv.vocab)
    random.shuffle(vocab) # randomly shuffle
    if num_pos+num_neg<len(vocab):
        pseudo_pos_list = vocab[:num_pos]
        pseudo_neg_list = vocab[-num_neg:]
        build_dict(pseudo_pos_list, "wv\\{}".format(pos_dict_name))
        build_dict(pseudo_neg_list, "wv\\{}".format(neg_dict_name))

    return pseudo_pos_list,pseudo_neg_list

def drop_duplicates(wordlist1, wordlist2):
    """
    update Dec.25, 2021
    drop duplicated words in two (contradictory) dicts
    :param wordlist1: word lists, list
    :param wordlist2: word lists, list
    :return: new_wordlist1, new_wordlist2, counts of duplicated words
    """
    word1 = set(wordlist1)
    word2 = set(wordlist2)
    word_common = word1 & word2
    count = len(word_common)
    new_wordlist1 = list(word1- word_common)
    new_wordlist2 = list(word2 - word_common)
    print("# words in wordlist1: {}".format(len(wordlist1)))
    print("# words in wordlist2: {}".format(len(wordlist2)))
    print("# duplicates: {}".format(count))
    print("# words in new wordlist1: {}".format(len(new_wordlist1)))
    print("# words in new wordlist2: {}".format(len(new_wordlist2)))
    return new_wordlist1, new_wordlist2

def build_keyword_dict(model, keywords,topn,filename,nrounds=30,threshold=0.0):
    """
    update Nov.20, 2021
    build dictionaries based on keywords
    :param model: trained w2v model
    :param keywords: list of keywords (for n_gram models, n>=3), list
    :param topn: find topn most similar words of every word in the init_list
    :param filename: filename of the saved dict, string
    :param :nrounds: rounds of recursion, int
    :param :threshold: threshold of cosine similarity, float
    :return: list of words
    """
    # generation of keywords list
    keyword_list = []
    vocab = list(model.wv.vocab)
    for i in vocab:
        for k in keywords:
            if k in i:
                keyword_list.append(i)

    build_dict(keyword_list, filename + "_key") # find n_gram phrases that contain keywords
    #print(keyword_list)
    add_words_list = find_words(model, keyword_list, topn, nrounds, threshold)
    print(len(add_words_list))

    #build_dict(add_words_list, filename) # expand the initial keywords list using w2v
    return add_words_list

def calc_cosine_SO(model, Pwords, Nwords,vocab):
    """
    Feb, 2022
    calculate semantic orientation from cosine similarity learned by the w2v model
    :param model: the w2v model
    :param Pwords: list of pre-defined positive words
    :param Nwords: list of neg-defined positive words
    :param vocab: vocabulary (model.wv.vocab)
    :return: standarised SO of words in the vocabulary (zscore)
    """

    dict = {}
    Pword = list(set(Pwords)&set(vocab)) # set of positive words in the w2v vocabulary
    Nword = list(set(Nwords)&set(vocab))  # set of positive words in the w2v vocabulary
    print("num of Pwords:{}".format(len(Pword)))
    print("num of Nwords:{}".format(len(Nword)))
    for i in vocab:
        association_pos = 0
        association_neg = 0
        for pword in Pword:
            association_pos += model.similarity(i,pword)
        for nword in Nword:
            association_neg += model.similarity(i,nword)
        dict[i] = association_pos - association_neg
    word_list = list(dict.keys())
    SO = list(dict.values())
    SO_df =pd.DataFrame({"word":word_list,"SO_cos":SO})
    SO_df.SO_cos = zscore(SO_df.SO_cos)
    def word_map(x):
        if x in Pword:
            return 1
        if x in Nword:
            return -1
        else:
            return 0
    SO_df["seed_word"] = SO_df["word"].apply(lambda x:word_map(x))
    return SO_df



if __name__ == '__main__':
    """
    corpus = pkl.load(open("data\\MPR.pkl", "rb"))
    text = corpus["text"].sum()

    # load user-defined dictionaries
    user_dicts = os.listdir("dict")
    for i in user_dicts:
        if 'txt' in i:
            path = "dict\\"+i
            jieba.load_userdict(path)

    # text preprocessing
    cut_text = clean_text(text)
    #cut_text = gen_trigram(cut_text, 1, 1)
    #print(cut_text)

    # train the skip-gram model
    model = Word2Vec(cut_text,sg=1, min_count=10,window=5,size=200,workers=4)
    model = Word2Vec.load("out\\models\\word2vec_MPR.model")
    model.save("out\\models\\word2vec_MPR.model")
    vocab = model.wv.vocab # 4673

    # calculate SO from cosine similarity
    begin1 = time.clock()
    LM_w2v_SO = calc_cosine_SO(model,Pwords,Nwords) # Chinese LM dictionary 321 pos, 249 neg
    end1 = time.clock()
    print("{} seconds has passed".format(end1-begin1))
    begin2 = time.clock()
    INF_w2v_SO = calc_cosine_SO(model,Pwords2,Nwords2) # Inflation dictionary 13 pos, 1 neg
    end2 = time.clock()
    print("{} seconds has passed".format(end2-begin2))

    LM_w2v_SO.to_csv("out\\results\\LM_w2v_SO.csv",encoding='utf-8-sig')
    INF_w2v_SO.to_csv("out\\results\\INF_w2v_SO.csv",encoding='utf-8-sig')

    ### alternative
    """

