import pickle as pkl
import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import pandas as pd
import matplotlib
import re
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns

"""
PENG Limin, 12th March, 2022
"""
"""
# global settings
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
os.chdir('D:\\PLM\\宏观金融seminar\\Seminar2021-2022\\央行沟通词典\\code')

# load user defined dictionaries as word lists, globally
stopwords = [line.strip() for line in open(r'dict\stopwords.txt', 'r', encoding='utf-8-sig').readlines()]  # 停用词词典
Pwords = [line.strip() for line in open(r'dict\LM_pos.txt', 'r', encoding='utf-8-sig').readlines()]
Nwords = [line.strip() for line in open(r'dict\LM_neg.txt', 'r', encoding='utf-8-sig').readlines()]
Pwords2 = [line.strip() for line in open(r'dict\INF_pos.txt', 'r', encoding='utf-8-sig').readlines()] # alternative lexicon
Nwords2 = [line.strip() for line in open(r'dict\INF_neg.txt', 'r', encoding='utf-8-sig').readlines()] # alternative lexicon
Pwords3 = [line.strip() for line in open(r'dict\pbc_N.txt', 'r', encoding='utf-8-sig').readlines()] # alternative lexicon
Nwords3 = [line.strip() for line in open(r'dict\pbc_P.txt', 'r', encoding='utf-8-sig').readlines()] # alternative lexicon
degreeDict = [line.strip() for line in open(r'dict\degreeDict.txt', 'r', encoding='utf-8-sig').readlines()] #
notDict = [line.strip() for line in open(r'dict\notDict.txt', 'r', encoding='utf-8-sig').readlines()]
stopwords = list(set(stopwords)-set(Pwords)-set(Nwords)-set(degreeDict)-set(notDict)-set(Pwords2)-set(Nwords2)-set(Pwords3)-set(Nwords3)) # 去重后的停用词

# load user defined dictionaries in jieba
user_dicts = os.listdir("dict")
user_dicts = [d for d in user_dicts if ".txt" in d]
for i in user_dicts:
    path = "dict\\" + i
    jieba.load_userdict(path)
"""
def clean_text_PPMI(text):
    """
    cut a single doc
    :param text: a single doc
    :return: parsed doc
    """
    stopwords = [line.strip() for line in open(r'dict\stopwords.txt', 'r', encoding='utf-8-sig').readlines()]
    text_ = ''.join(''.join(text.split('\n')).split('\x0c')) #去掉换行符换页符
    res = re.compile("[^\u4e00-\u9fa5^a-z^A-Z]")  # 去掉其他,仅保留中英文
    text_ = res.sub("", text_)
    word_list = jieba.cut(text_,cut_all=False,HMM=True)
    word_list = [w for w in word_list if w not in stopwords and len(w)>1]
    word_list = " ".join(word_list)
    return word_list

def build_dtm(corpus_cut,filename):
    """
    build the doc-term matrix using sklearn.feature_extraction.text (CountVectorizer & TfidfTransformer),
    save dtm(csv) & tf-idf dtm(csv) & vocab(list, unique words in the corpus) & net_sentiment(list, doc level net sentiment)
    :param corpus_cut: the given PARSED corpus, list of docs
    :param filename: name to save the file(matrix)
    """

    vectorizer=CountVectorizer(stop_words=stopwords) # 定义CountVectorizer类的对象
    X=vectorizer.fit_transform(corpus_cut)
    dtm = X.toarray() # 词频矩阵 (n_doc, V),doc-term matrix
    transformer=TfidfTransformer() # 定义TfidfTransformer类的对象
    tfidf=transformer.fit_transform(X) # 计算tf-idf
    vocab=vectorizer.get_feature_names() # 获取词袋模型中的所有词语  16527
    weight=tfidf.toarray() #td-idf加权后的词频矩阵
    dtm_df = pd.DataFrame(data=dtm,columns=vocab)
    tfidf_df = pd.DataFrame(data=weight,columns=vocab)
    dtm_df.to_csv("out\\results\\{}.csv".format(filename),encoding='utf-8-sig')
    tfidf_df.to_csv("out\\results\\{}tfidf_df.csv".format(filename),encoding='utf-8-sig')
    """
    Pwords = list(set(vocab)&set(Pwords))
    pos_dtm_df = dtm_df[Pwords]

    Nwords = list(set(vocab)&set(Nwords))
    neg_dtm_df = dtm_df[Nwords]

    pos_dtm_df['pos_sum'] = pos_dtm_df.apply(lambda x:x.sum(),axis=1)
    pos_dtm_df.loc['sum'] = pos_dtm_df.apply(lambda x:x.sum())
    neg_dtm_df['neg_sum'] = neg_dtm_df.apply(lambda x:x.sum(),axis=1)
    neg_dtm_df.loc['sum'] = neg_dtm_df.apply(lambda x:x.sum())

    pos_dtm_df.to_csv("out\\results\\pos_dtm_df.csv",encoding='utf-8-sig')
    neg_dtm_df.to_csv("out\\results\\neg_dtm_df.csv",encoding='utf-8-sig')
    net_sentiment = pos_dtm_df['pos_sum'][:-1]-neg_dtm_df['neg_sum'][:-1]
    pkl.dump(vocab,open("data\\vocab_bow.pkl","wb"))
    pkl.dump(net_sentiment,open("data\\net_sentiment_bow.pkl","wb"))
    #plt.plot(corpus["date"],net_sentiment.rolling(window=3).mean())
    #plt.plot(pos_dtm_df.loc['sum'][:-1].sort_values(ascending=False)[:50])
    #plt.xticks(pd.date_range('2001-03-01', '2021-07-07', freq='6m'), rotation=45)

    print(list(set(vocab)&set(degreeDict))) # []
    print(list(set(vocab)&set(notDict))) #  ['不是', '不要', '难以', '并不', '没有', '并无', '不能']
    """

def build_sentence_df(corpus,filename):
    sentence_list = []
    num_s_list = []
    date_list_expand = []
    date_list = corpus.date.to_list()

    def split_sentence(x,sentence_list):
        sentence =  re.split(r'(\!|\?|。|！|？)', x) # 短句
        #res = re.compile("[^\u4e00-\u9fa5^a-z^A-Z]")  # 去掉其他,仅保留中英文
        sentence = [s for s in sentence if len(s)>1]
        num_s = len(sentence)
        sentence_list += sentence
        num_s_list.append(num_s)

    for i in range(len(date_list)):
        split_sentence(corpus['text'][i],sentence_list)
        duplicate = [date_list[i]]*num_s_list[i]
        date_list_expand += duplicate

    sentence_df = pd.DataFrame({'date':date_list_expand,'sentence':sentence_list})
    sentence_df['len'] = sentence_df['sentence'].apply(lambda x:len(x))
    #doc_len = sentence_df.groupby('date')['len'].sum()
    corpus['len'] = corpus['text'].apply(lambda x:len(x))
    sentence_df["cut"] = sentence_df["sentence"].apply(lambda x:clean_text_PPMI(x))
    sentence_df.to_csv("data\\{}.csv".format(filename),encoding='utf-8-sig')
    pkl.dump(sentence_df,open("data\\{}.pkl".format(filename),"wb"))

def build_ttm(sentence,posword,negword,vocab,ttm):
    """
    build the term-term co-occurrence matrix, at the sentence level (i.e., only consider co-occurrence in a sentence)
    :param sentence: with the given PARSED sentence labelled as "cut", row of pandas df
    :param posword: list of positive words, list
    :param negword: list of negative words, list
    :param vocab: vocabulary
    :param ttm: initial
    :return: tuples of term counts (pos neg )
    """
    word_list = sentence["ngram"] # parse the sentence into list of words
    word_list = [w for w in word_list if w in vocab and w not in stopwords and len(w)>1]
    pos_list = [w for w in word_list if w in posword]
    neg_list = [w for w in word_list if w in negword]
    No_list = [w for w in word_list if w in notDict]
    num_pos = len(pos_list)
    num_neg = len(neg_list)
    num_no = len(No_list)
    if num_pos & num_neg==0 & num_no==0:
        for p in pos_list:
            ind_p = vocab.index(p)
            for w in word_list:
                ind_w = vocab.index(w)
                ttm[ind_p,ind_w] += 1
                ttm[ind_w,ind_p] += 1
    if num_neg & num_pos==0 & num_no==0:
        for n in neg_list:
            ind_n = vocab.index(n)
            for w in word_list:
                ind_w = vocab.index(w)
                ttm[ind_n,ind_w] += 1
                ttm[ind_w,ind_n] += 1
    return (num_neg, num_pos, num_no)

def change_dt(dtime):
    if dtime.month == 3 or dtime.month == 12:
        new_dtime = dtime+datetime.timedelta(days=30)
    else:
        new_dtime = dtime + datetime.timedelta(days=29)
    return new_dtime

def map_word(w):
    cat=0
    for i in range(5):
        if w in word_merged[i]:
            cat = i + 1
            break
    return cat

def isin_lexicon(w):
    if w==1:
        return "positive"
    if w==-1:
        return "negative"
    else:
        return "out of lexicon"

def zscore(arr):
    """
    normalized values of the given array
    :param arr: array
    :return: normalized values of the given array
    """
    mu = arr.mean()
    sigma = arr.std()
    return (arr - mu) / sigma

def calc_PPMI_SO(ttm,vocab,Pwords,Nwords,min_count=6):
    Pwords_ = list(set(Pwords)&set(vocab))
    Nwords_ = list(set(Nwords)&set(vocab))
    PPMI_SO = []
    word_list = []
    for v in vocab:
        ind_v = vocab.index(v)
        f_v = ttm[ind_v,:].sum() # freq of term_v
        PPMI_P = 0 # sum of PPMI_vp (v with all pos words)
        PPMI_N = 0 # sum of PPMI_vn (v with all neg words)
        if f_v>=min_count: # only consider the case that f_v is above threshold in the ttm
            word_list.append(v)
            for p in Pwords_:
                ind_p = vocab.index(p)
                f_vp = ttm[ind_v,ind_p] # freq of term_v and term_p (co-occur)
                if f_vp:
                    f_p = ttm[:,ind_p].sum() # freq of term_p (positive word)
                    PMI_vp = np.log2(f_vp/(f_p*f_v))+np.log2(45026) # |S|=45026
                    PPMI_vp = max(PMI_vp, 0)
                    PPMI_P += PPMI_vp
            for n in Nwords_:
                ind_n = vocab.index(n)
                f_vn = ttm[ind_v,ind_n] # freq of term_v and term_n (co-occur)
                if f_vn:
                    f_n = ttm[:, ind_n].sum()  # freq of term_n (negative word)
                    PMI_vn = np.log2(f_vn/(f_n*f_v))+np.log2(45026) # |S|=45026
                    PPMI_vn = max(PMI_vn, 0)
                    PPMI_N += PPMI_vn
            PPMI_SO_v = PPMI_P - PPMI_N
            PPMI_SO.append(PPMI_SO_v)
        else:
            continue
    PPMI_SO_df = pd.DataFrame({"word":word_list,"SO_PPMI":PPMI_SO})
    PPMI_SO_df["SO_PPMI"] = zscore(PPMI_SO_df["SO_PPMI"])
    def word_map(x):
        if x in Pwords_:
            return 1
        if x in Nwords_:
            return -1
        else:
            return 0
    PPMI_SO_df["seed_word"] = PPMI_SO_df["word"].apply(lambda x:word_map(x))
    return PPMI_SO_df

def sum_stat_SO(df,label):
    word_list = []
    num_pos = []
    num_neg = []
    num_words = []
    means = []
    mins = []
    maxs = []
    q25 = df[df[label]<=df[label].quantile(0.25)].sort_values(by=label)
    q50 = df[(df[label] > df[label].quantile(0.25)) & (df[label] <= df[label].quantile(0.5))].sort_values(
        by=label)
    q75 = df[(df[label] > df[label].quantile(0.5)) & (df[label] <= df[label].quantile(0.75))].sort_values(
        by=label)
    q100 = df[(df[label] > df[label].quantile(0.75)) & (df[label] <= df[label].quantile(1))].sort_values(
        by=label)
    q_df_list = [q25, q50, q75, q100]
    for q_df in q_df_list:
        q_words = list(q_df["word"])
        num_pos.append(len(q_df[q_df['seed_word_LM']==1]))
        num_neg.append(len(q_df[q_df['seed_word_LM']==-1]))
        num_words.append(len(q_df))
        means.append(q_df[label].mean())
        mins.append(q_df[label].min())
        maxs.append(q_df[label].max())
        word_list.append(q_words)
    sum_df = pd.DataFrame({"mean":means,"min":mins,"max":maxs,"num_seed_pos":num_pos,"num_seed_neg":num_neg,
                          "num_words":num_words})
    sum_df.to_csv("out\\results\\sum_{}.csv".format(label),encoding='utf-8-sig')
    return word_list

def label_quantile(df,label,x):
    """
    label the quantile category of SO scores (1: <Q1,2:Q1-median, 3:median-Q3,4:>Q3)
    :param df: pandas dataframe, containing a column of SO score
    :param label: string, name of the columns of SO score (SO_cos, SO_PPMI)
    :return: int, quantile category
    """
    q25 = df[label].quantile(0.25)
    q50 = df[label].quantile(0.50)
    q75 = df[label].quantile(0.75)
    if x<=q25:
        return 1
    elif x<q50:
        return 2
    elif x<q75:
        return 3
    else:
        return 4

def label_LM_quantile(df,label,x,lb=0.25,ub=0.75):
    """
    label the quantile category of SO scores (1: <Q1,2:Q1-median, 3:median-Q3,4:>Q3)
    :param df: pandas dataframe, containing a column of SO score
    :param label: string, name of the columns of SO score (SO_cos, SO_PPMI)
    :return: int, quantile category
    """
    Q1 = df[df["seed_word"]==-1][label].quantile(lb)
    Q3 = df[df["seed_word"] == 1][label].quantile(ub)
    if x<=Q1:
        return -1 # labeled as negative
    elif x>=Q3:
        return 1 # labeled as positive
    else:
        return 0

"""
all_SO["label_cos"] = all_SO["SO_cos"].apply(lambda x:label_quantile(all_SO,"SO_cos",x))
all_SO["label_PPMI"] = all_SO["SO_PPMI"].apply(lambda x:label_quantile(all_SO,"SO_PPMI",x))
domestic_SO["label_cos"] = domestic_SO["SO_cos"].apply(lambda x:label_quantile(all_SO,"SO_cos",x))
domestic_SO["label_PPMI"] = domestic_SO["SO_PPMI"].apply(lambda x:label_quantile(all_SO,"SO_PPMI",x))
all_SO["selected"] = all_SO["label_cos"]+all_SO["label_PPMI"]
domestic_SO["selected"] = domestic_SO["label_cos"]+domestic_SO["label_PPMI"]
all_SO.to_csv("out\\results\\all_SO.csv",encoding='utf-8-sig')
domestic_SO.to_csv("out\\results\\domestic_SO.csv",encoding='utf-8-sig')
all_SO = pd.read_csv("out\\results\\all_SO.csv",index_col=0)
domestic_SO = pd.read_csv("out\\results\\domestic_SO.csv",index_col=0)
pd.merge(all_SO[all_SO["selected"]==2],domestic_SO[domestic_SO["selected"]==2],on="word",how="inner").to_csv(
    "out\\results\\merged_neg.csv",encoding="utf-8-sig")
pd.merge(all_SO[all_SO["selected"]==8],domestic_SO[domestic_SO["selected"]==8],on="word",how="inner").to_csv(
    "out\\results\\merged_pos.csv",encoding="utf-8-sig")

domestic_LM_selected = {}
domestic_LM_cos = {}
domestic_LM_PPMI = {}
domestic_LM_sel = domestic_SO[(domestic_SO["seed_word"]!=0)&((domestic_SO["LM_selected"]==-2)|(domestic_SO["LM_selected"]==2))]
domestic_LM_sel.reset_index(inplace=True)
neg_count = 0
pos_count = 0
for i in range(len(domestic_LM_sel)):
    word = domestic_LM_sel.loc[i,:]["word"]
    SO_cos = domestic_LM_sel.loc[i,:]["SO_cos"]
    SO_PPMI = domestic_LM_sel.loc[i, :]["SO_PPMI"]
    indicator = 1
    pos_count += 1
    if domestic_LM_sel.loc[i,:]["seed_word"] == -1:
        indicator = -1
        neg_count += 1
        pos_count -= 1
    domestic_LM_selected[word] = indicator
    domestic_LM_cos[word] = SO_cos
    domestic_LM_PPMI[word] = SO_PPMI
pkl.dump(domestic_LM_selected,open("data\\weight_domesticLM_sel.pkl","wb"))
pkl.dump(domestic_LM_cos,open("data\\weight_domesticLM_sel_cos.pkl","wb"))
pkl.dump(domestic_LM_PPMI,open("data\\weight_domesticLM_sel_PPMI.pkl","wb"))

all_SO = pd.read_csv("out\\results\\all_SO.csv",index_col=0)
domestic_SO = pd.read_csv("out\\results\\domestic_SO.csv",index_col=0)
all_SO["label_cos_LM"] = all_SO["SO_cos"].apply(lambda x:label_LM_quantile(all_SO,"SO_cos",x))
all_SO["label_PPMI_LM"] = all_SO["SO_PPMI"].apply(lambda x:label_LM_quantile(all_SO,"SO_PPMI",x))
all_SO["LM_selected"] = all_SO["label_cos_LM"]+all_SO["label_PPMI_LM"]
domestic_SO["label_cos_LM"] = domestic_SO["SO_cos"].apply(lambda x:label_LM_quantile(domestic_SO,"SO_cos",x))
domestic_SO["label_PPMI_LM"] = domestic_SO["SO_PPMI"].apply(lambda x:label_LM_quantile(domestic_SO,"SO_PPMI",x))
domestic_SO["LM_selected"] = domestic_SO["label_cos_LM"]+domestic_SO["label_PPMI_LM"]
all_SO.to_csv("all_SO.csv",encoding='utf-8-sig')
domestic_SO.to_csv("domestic_SO.csv",encoding='utf-8-sig')

if __name__ == '__main__':
""""""
    corpus = pkl.load(open("data\\MPR0227.pkl", "rb"))
    sentence_df = pkl.load(open("data\\sentence_df.pkl", "rb"))

    drop_ind = sentence_df[sentence_df["len"] > sentence_df["len"].quantile(0.995)].index.to_list()
    sentence_df2 = sentence_df.drop(drop_ind,axis=0)
    drop_ind2 = sentence_df[sentence_df["len"] < 5].index.to_list()
    sentence_df2 = sentence_df2.drop(drop_ind2, axis=0)
    sentence_df2["num_word"] = sentence_df2["cut"].apply(lambda x:len(x.split()))
    drop_ind3 = sentence_df2[sentence_df2["num_word"]<5].index.to_list()
    sentence_df2.drop(drop_ind3,axis=0, inplace=True)
    drop_ind4 = sentence_df2[sentence_df2["len"] > 385].index.to_list()
    sentence_df2.drop(drop_ind4, axis=0, inplace=True)
    sentence_df2 = sentence_df2.reset_index()
    sentence_df2.drop(['index'],axis=1,inplace=True)
    drop_ind5 = sentence_df2[sentence_df2["len"] > 300].index.to_list()
    sentence_df2.drop(drop_ind5, axis=0, inplace=True)
    sentence_df2.to_csv("data\\sentence_df.csv",encoding='utf-8-sig')
    pkl.dump(sentence_df2, open("data\\sentence_df.pkl", "wb"))
       
    # build doc-term co-occurrence matrix
    corpus["cut"] = corpus["text"].apply(lambda x: clean_text(x))
    corpus_cut = corpus["cut"].tolist()  # list of docs
    buid_dtm(corpus_cut)
 
    # build term-term co-occurrence matrix
    vocab = pkl.load(open("data\\vocab_merged.pkl","rb"))
    
    model = Word2Vec.load("out\\models\\word2vec_MPR.model")
    vocab_wv =list(model.wv.vocab)
    vocab = list(set(vocab)&set(vocab_wv))
    vocab = pkl.dump(vocab,open("data\\vocab_merged.pkl","wb"))
    
    Pwords = list(set(vocab)&set(Pwords)) #321
    Nwords = list(set(vocab) & set(Nwords)) #249
    V = len(vocab)
    ttm = np.zeros(shape=(V, V))
    # generate sentence-level summary statistics & calculate PMI-SO
    begin1 = time.clock()
    sentence_df[["num_neg", "num_pos", "num_no"]] = sentence_df.apply(lambda x: build_ttm(x, Pwords, Nwords), axis=1, 
                                                                        result_type='expand')
    end1 = time.clock()
    print("{} seconds has passed".format(end1 - begin1))
    pkl.dump(ttm,open("data\\ttm0228.pkl","wb"))

    # calculate SO from PPMI
    vocab = pkl.load(open("data\\vocab_merged.pkl","rb")) # the saved vocabulary
    ttm = pkl.load(open("data\\ttm0228.pkl","rb"))
    begin1 = time.clock()
    LM_PPMI_SO = calc_PPMI_SO(ttm, vocab, Pwords, Nwords)  # Chinese LM dictionary 321 pos, 249 neg
    end1 = time.clock()
    print("{} seconds has passed".format(end1 - begin1)) # 9032.0531595 seconds has passed
    LM_PPMI_SO.to_csv("out\\results\\LM_PPMI_SO.csv", encoding='utf-8-sig')
    LM_w2v_SO = pd.read_csv("out\\results\\LM_w2v_SO.csv", encoding='utf-8-sig',index_col=0)
    LM_PPMI_SO = pd.read_csv("out\\results\\LM_PPMI_SO.csv", encoding='utf-8-sig',index_col=0)

    SO_df = pd.merge(LM_w2v_SO,LM_PPMI_SO,how = 'inner', on = 'word')
    SO_df[['SO_sim','SO_PPMI']].corr(method='pearson') # pearson:0.412697, spearman:0.404108
    SO_df.to_csv("out\\results\\SO_df.csv", encoding='utf-8-sig')
    sns.set_theme(color_codes=True)
    sns.regplot(data=SO_df[['SO_sim', 'SO_PPMI']], x='SO_sim', y='SO_PPMI',scatter_kws={'alpha': 0.1,'color':'b','s':10}
                ,line_kws={'alpha': 0.5,'color': 'r'},ci=99)
    plt.savefig('out\\figs\\regplot.jpg',dpi=600)
    plt.show()
    plt.hist(SO_df['SO_PPMI'],bins=100)
    print(SO_df["SO_PPMI"].mode())
    SO_df[SO_df["SO_PPMI"]==SO_df["SO_PPMI"].mode()]["word"]
    pkl.dump(SO_df,open("out\\results\\SO_df.pkl","wb"))

    # summary stat of SO scores
    word_sim = sum_stat_SO(SO_df,"SO_sim")
    word_PPMI = sum_stat_SO(SO_df, "SO_PPMI")
    # compare every single word's SO
    word_merged = []
    for i in range(5):
        k=2*i
        word_merged.append(list(set(word_sim[k]+word_sim[k+1])&set(word_PPMI[k]+word_PPMI[k+1])))
    for i in range(5):
        print(len(word_merged[i]),word_merged[i])
    pkl.dump(word_merged,open("out\\results\\word_merged.pkl","wb")

    SO_df["rank"] = SO_df["word"].map(map_word)
    SO_df["isin_lexicon"] = SO_df['seed_word_LM'].apply(isin_lexicon)
    pkl.dump(SO_df, open("out\\results\\SO_df.pkl", "wb"))
    SO_df.to_csv("out\\results\\SO_df.csv",encoding='utf-8-sig')
    SO_df2 = SO_df[(SO_df["rank"]==1) | (SO_df["rank"]==5)]
    SO_df2["isin_lexicon"] = SO_df2['seed_word_LM'].apply(isin_lexicon)
    pkl.dump(SO_df2, open("out\\results\\SO_df_merged.pkl", "wb"))
    SO_df2.to_csv("out\\results\\SO_df_merged.csv", encoding='utf-8-sig')

    #sns.set_theme(color_codes=True)
    SO_df2 = pkl.load(open("out\\results\\SO_df_merged.pkl", "rb"))
    SO_df = pkl.load(open("out\\results\\SO_df.pkl", "rb"))
    sns.scatterplot(data=SO_df2[(SO_df2["rank"]==1)|(SO_df2["rank"]==5)], x='SO_cos', y='SO_PPMI',hue='isin_lexicon',alpha=0.5, hue_order=['positive','negative',"out of lexicon"],
                    palette={'positive':'r',"negative":'g',"out of lexicon":"b"})
    plt.xlim(-3, 3)
    plt.ylim(-6, 6)
    plt.savefig('out\\figs\\LM_SO2.jpg', dpi=600)
    plt.show()

    sns.set_theme(color_codes=True)
    sns.set_palette("muted")
    SO_df_LM = SO_df[SO_df["isin_lexicon"]!="out of lexicon"]
    sns.scatterplot(data=SO_df_LM, x='SO_cos', y='SO_PPMI', hue='isin_lexicon', hue_order=['positive','negative'],
                    alpha=0.5,palette={'positive':'r',"negative":'g'})
    plt.xlim(-3, 3)
    plt.ylim(-6, 6)
    plt.savefig('out\\figs\\LM_SO.jpg', dpi=600)
    plt.show()
    """