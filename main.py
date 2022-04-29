import pickle as pkl
import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import pandas as pd
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import re
import time
import numpy as np
import seaborn as sns
from PMI_SO import *
from w2v_SO import *
from sentence_sent_score import *
from ngram import *

"""
PENG Limin, 12th March, 2022
"""

# global settings, should not be changed
os.chdir('D:\\PLM\\宏观金融seminar\\Seminar2021-2022\\央行沟通词典\\code')

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
#sns.set_style('white')
#sns.set_palette("muted")

# load user defined dictionaries as word lists, globally
stopwords = [line.strip() for line in open(r'dict\stopwords.txt', 'r', encoding='utf-8-sig').readlines()]  # 停用词词典
#P = [line.strip() for line in open(r'dict\pbc_P.txt', 'r', encoding='utf-8-sig').readlines()]
#N = [line.strip() for line in open(r'dict\pbc_N.txt', 'r', encoding='utf-8-sig').readlines()]
#P = [line.strip() for line in open(r'dict\INF_pos.txt', 'r', encoding='utf-8-sig').readlines()]
#N = [line.strip() for line in open(r'dict\INF_neg.txt', 'r', encoding='utf-8-sig').readlines()]
P = [line.strip() for line in open(r'dict\LM_pos.txt', 'r', encoding='utf-8-sig').readlines()]
N = [line.strip() for line in open(r'dict\LM_neg.txt', 'r', encoding='utf-8-sig').readlines()]
#degreeDict = [line.strip() for line in open(r'dict\degreeDict.txt', 'r', encoding='utf-8-sig').readlines()] #
notDict = [line.strip() for line in open(r'dict\notDict.txt', 'r', encoding='utf-8-sig').readlines()]
#degreeDict = list(set(degreeDict)-set(P)-set(N))
notDict = list(set(notDict)-set(P)-set(N))
stopwords = list(set(stopwords)-set(P)-set(N)-set(notDict)) # 去重后的停用词

# load user defined dictionaries in jieba
user_dicts = os.listdir("dict")
user_dicts = [d for d in user_dicts if (".txt" in d) and ('notDict' not in d) and ('stopwords' not in d)] # do not load stopwords
for i in user_dicts:
    path = "dict\\" + i
    jieba.load_userdict(path)

def gen_SO_df(df,filename,P,N,min_count=6):
    """
    Calculate the cosine SO (based on w2v embeddings) and the PPMI SO
    :param df: pandas dataframe
    :param filename: filename to save the result
    :param P: positive word lists
    :param N: negative word list
    :return: saved dataframe of SO results
    """
    w2v_ = df["ngram"].to_list()
    model_ = Word2Vec(w2v_,sg=1, min_count=min_count, window=20,size=200,workers=4)
    vocab_ = list(model_.wv.vocab)
    cos_SO = calc_cosine_SO(model_,P,N,vocab_)
    cos_SO = zscore(cos_SO["cos_SO"])
    model_.save("out\\models\\model_{}.model".format(filename))
    V_ = len(vocab_)
    ttm_ = np.zeros(shape=(V_, V_))
    df.apply(lambda x: build_ttm(x,P,N,vocab_,ttm_), axis=1,result_type='expand')
    PPMI_SO = calc_PPMI_SO(ttm_,vocab_,P,N)
    #pkl.dump(ttm_,open("out\\results\\ttm_{}.pkl".format(filename),"wb"))
    SO_df_ = pd.merge(cos_SO,PPMI_SO,on='word',how='inner')
    SO_df_.drop(columns='seed_word_x',inplace=True)
    SO_df_.rename(columns={'seed_word_y':'seed_word'},inplace=True)
    SO_df_.to_csv("out\\results\\SO_df_{}.csv".format(filename),encoding='utf-8-sig')

MPR_all = pkl.load(open("data\\MPR_all.pkl","rb"))
MPR_domestic = pkl.load(open("data\\MPR_domestic.pkl","rb"))
MPR_result = pkl.load(open("out\\results\\MPR_result.pkl","rb"))
build_sentence_df(MPR_all,"sentence_df_all")
build_sentence_df(MPR_domestic,"sentence_df_domestic")
sentence_df_all = pkl.load(open("out\\results\\sentence_df_all.pkl","rb")) # build_sentence_df(MPR_all,"sentence_df_all")
sentence_df_domestic = pkl.load(open("out\\results\\sentence_df_domestic.pkl","rb")) # build_sentence_df(MPR_domestic,"sentence_df_domestic")
sentence_df_all["num_word"] = sentence_df_all["cut"].apply(lambda x:len(x.split()))
sentence_df_all.drop(sentence_df_all[sentence_df_all["num_word"]<=1].index,inplace=True)
sentence_df_domestic["num_word"] = sentence_df_domestic["cut"].apply(lambda x:len(x.split()))
sentence_df_domestic.drop(sentence_df_domestic[sentence_df_domestic["num_word"]<=1].index,inplace=True)
sentence_df_all["ngram"] = sentence_df_all["sentence"].apply(lambda x:gen_ngram(x,5))
sentence_df_domestic["ngram"] = sentence_df_domestic["sentence"].apply(lambda x:gen_ngram(x,5))

sentence_df_all.drop(['LM_sel', 'LM_sel_cos', 'LM_sel_PPMI'], axis = 1, inplace=True)
sentence_df_domestic.drop(['LM_sel', 'LM_sel_cos', 'LM_sel_PPMI'], axis = 1, inplace=True)
sentence_df_domestic["label"] = sentence_df_domestic["sentence"].apply(lambda x:label_sentence(x))
# summary statistics
sentence_df_domestic_sel = pkl.load(open("out\\results\\sentence_df_domestic_sel.pkl","rb"))
sentence_df_domestic_sel.drop(['LM','label2','inflation', 'credit', 'labor', 'output', 'MP',
       'stock', 'bond', 'adj', 'external','ChFin_inflation',
       'ChFin_credit', 'ChFin_labor', 'ChFin_output', 'ChFin_MP',
       'ChFin_stock', 'ChFin_bond'], axis = 1, inplace=True)
sentence_df_domestic_sel["label"] = sentence_df_domestic_sel["sentence"].apply(lambda x:label_sentence(x))
v_all = sentence_df_all["cut"].sum().split()
v_domestic = sentence_df_domestic_sel["cut"].sum().split()
print("num of vocab (all): ", len(list(set(v_all))))
print("num of P words (all): ",len(list(set(v_all)&set(P))))
print("num of N words (all): ",len(list(set(v_all)&set(N))))
print("num of vocab (domestic): ", len(list(set(v_domestic))))
print("num of P words (domestic): ",len(list(set(v_domestic)&set(P))))
print("num of N words (domestic): ",len(list(set(v_domestic)&set(N))))
print(sentence_df_all["num_word"].mean(),sentence_df_all["num_word"].median(),sentence_df_all["num_word"].min(),sentence_df_all["num_word"].max())
print(sentence_df_all["len"].mean(),sentence_df_all["len"].median(),sentence_df_all["len"].min(),sentence_df_all["len"].max())
## fig1 MPR number of words by quarter
plt.figure(figsize=(16, 9))
plt.grid(axis="y")
ts=MPR_all["quarter"]
plt.plot(ts,MPR_all["len"],'k', linestyle='-', marker='o',label="本季度值")
plt.plot(ts,MPR_all["len"].rolling(window=3).mean().values,'grey', linestyle='--', marker='.',label="三季度移动平均")
plt.xticks(range(0,len(ts),2),rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper right',fontsize=15)
plt.savefig('out\\figs\\document_len.png', dpi=600)
## fig2 MPR average sentence_length by quarter
plt.figure(figsize=(16, 9))
plt.grid(axis="y")
mean_sentence = sentence_df_all.groupby("date")["len"].mean()
plt.plot(ts,mean_sentence,'k',linestyle='-', marker='o',label="本季度值")
plt.plot(ts,mean_sentence.rolling(window=3).mean().values,'grey', linestyle='--', marker='.',label="三季度移动平均")
plt.xticks(range(0,len(ts),2),rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper right',fontsize=15)
plt.savefig('out\\figs\\avg_sentence_len.png', dpi=600)
## fig3 ChFin overall
plt.figure(figsize=(16, 9))
ts = MPR_result["quarter"]
plt.plot(ts,MPR_result["ChFin_overall"],'k',linestyle='-', marker='o',label="ChFin_overall当季值")
plt.plot(ts,MPR_result["ChFin_overall"].rolling(window=3).mean().values,'grey', linestyle='--',
         marker='.',label="ChFin_overall三季度移动平均")
plt.xticks(range(0,len(ts),2),rotation=90,fontsize=15)
plt.ylim([-4,4])
plt.yticks(fontsize=15)
plt.legend(loc='upper right',fontsize=15)
plt.vlines('2001q3', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="911\n阿富汗战争",xy=('2001q3',-3.7),xytext=('2001q1',-3.7),fontsize=13)
plt.vlines('2001q4', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="加入WTO",xy=('2001q4',1),xytext=('2001q2',1),fontsize=13)
plt.vlines('2003q2', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="非典",xy=('2003q2',-2.3),xytext=('2003q1',-2.3),fontsize=13)
plt.vlines('2008q3', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="雷曼兄弟破产",xy=('2008q3',-1.8),xytext=('2008q1',-1.8),fontsize=13)
plt.vlines('2009q4', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="四万亿出台\n欧债危机",xy=('2009q4',1.5),xytext=('2009q2',1.5),fontsize=13)
plt.vlines('2011q1', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="日本311地震",xy=('2011q1',-1),xytext=('2010q1',-1),fontsize=13)
plt.vlines('2015q2', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="A股震荡",xy=('2015q2',0.7),xytext=('2014q4',0.7),fontsize=13)
plt.vlines('2016q2', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="英国脱欧\n中东危机",xy=('2016q2',-1.5),xytext=('2015q4',-1.5),fontsize=13)
plt.vlines('2018q1', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="中美贸易战",xy=('2018q1',0.2),xytext=('2017q3',0.2),fontsize=13)
plt.vlines('2020q1', -4, 4, colors='grey', linestyles=':')
plt.annotate(s="新冠疫情",xy=('2020q1',-1.5),xytext=('2019q3',-1.5),fontsize=13)
plt.savefig('out\\figs\\ChFin_overall.png', dpi=600)
# w2v & cosine similarity & lexical semantic orientation
gen_SO_df(sentence_df_domestic_sel,"domestic_ngram",P,N,min_count=6)
# adj
pbc_pos = [line.strip() for line in open(r'dict\alternative_dict\pbc_pos.txt', 'r', encoding='utf-8-sig').readlines()] #164
pbc_neg = [line.strip() for line in open(r'dict\alternative_dict\pbc_neg.txt', 'r', encoding='utf-8-sig').readlines()] #155
ChFin_pbc_pos = [line.strip() for line in open(r'dict\alternative_dict\ChFin_pbc_pos.txt', 'r', encoding='utf-8-sig').readlines()] #245
ChFin_pbc_neg = [line.strip() for line in open(r'dict\alternative_dict\ChFin_pbc_neg.txt', 'r', encoding='utf-8-sig').readlines()] #327
## calculate SO
bt = time.clock()
gen_SO_df(sentence_df_domestic_sel,"domestic_ngram_ChFinpbc3",ChFin_pbc_pos,ChFin_pbc_neg,min_count=6) # recaculate SO using the new seed word set
et = time.clock()
print(et-bt)
## calculate tone
ChFin_pbc_adj_pos = [line.strip() for line in open(r'dict\alternative_dict\ChFin_pbc_adj_pos.txt', 'r', encoding='utf-8-sig').readlines()] #228
ChFin_pbc_adj_neg = [line.strip() for line in open(r'dict\alternative_dict\ChFin_pbc_adj_neg.txt', 'r', encoding='utf-8-sig').readlines()] #290
print(len(ChFin_pbc_adj_pos),len(ChFin_pbc_adj_neg))
weight_pbc = build_weight_dict(pbc_pos,pbc_neg,"weight_pbc")
weight_ChFin_pbc = build_weight_dict(ChFin_pbc_pos,ChFin_pbc_neg,"weight_ChFin_pbc")
weight_ChFin_pbc_adj = build_weight_dict(ChFin_pbc_adj_pos,ChFin_pbc_adj_neg,"weight_ChFin_pbc_adj")
sentence_df_domestic_sel["pbc"] = sentence_df_domestic_sel.apply(lambda x:calc_SO(x,weight_pbc),axis=1)
sentence_df_domestic_sel["ChFin_pbc"] = sentence_df_domestic_sel.apply(lambda x:calc_SO(x,weight_ChFin_pbc),axis=1)
sentence_df_domestic_sel["ChFin_pbc_adj"] = sentence_df_domestic_sel.apply(lambda x:calc_SO(x,weight_ChFin_pbc_adj),axis=1)
for i in ["ChFin_pbc","ChFin_pbc_adj"]:
    MPR_result[i] = zscore(sentence_df_domestic_sel.groupby("date")[i].mean().values)
MPR_result[['GDP_g', 'M2_g', 'repo_rate', 'CPI', 'PPI', 'confidence','ChFin', 'pbc', 'ChFin_pbc',
            'ChFin_pbc_adj']].corr().to_csv("out\\results\\corr.csv",encoding='utf-8-sig')
# By Topic
sentence_sel_output = sentence_df_domestic_sel[sentence_df_domestic_sel["output"]==1]
sentence_sel_inflation = sentence_df_domestic_sel[sentence_df_domestic_sel["inflation"]==1]
sentence_sel_credit = sentence_df_domestic_sel[sentence_df_domestic_sel["credit"]==1]
sentence_sel_MP = sentence_df_domestic_sel[sentence_df_domestic_sel["MP"]==1]
sentence_sel_int = sentence_df_domestic_sel[sentence_df_domestic_sel["int_r"]==1]

"""
import ast
sentence_sel_output["ngram"] = sentence_sel_output["ngram"].apply(lambda x:ast.literal_eval(x))
"""
pkl.dump(sentence_sel_output,open("data\\sentence_sel_output.pkl","wb"))
pkl.dump(sentence_sel_inflation,open("data\\sentence_sel_inflation.pkl","wb"))
pkl.dump(sentence_sel_credit,open("data\\sentence_sel_credit.pkl","wb"))
pkl.dump(sentence_sel_MP,open("data\\sentence_sel_MP.pkl","wb"))
pkl.dump(sentence_sel_int,open("data\\sentence_sel_int.pkl","wb"))
sentence_sel_output = pkl.load(open("data\\sentence_sel_output.pkl","rb"))
sentence_sel_inflation = pkl.load(open("data\\sentence_sel_inflation.pkl","rb"))
sentence_sel_credit = pkl.load(open("data\\sentence_sel_credit.pkl","rb"))
sentence_sel_MP = pkl.load(open("data\\sentence_sel_MP.pkl","rb"))
sentence_sel_int = pkl.load(open("data\\sentence_sel_int.pkl","rb"))
gen_SO_df(sentence_sel_output,"output",P_adj,N_adj,min_count=4) #P282, N235, S6348
gen_SO_df(sentence_sel_inflation,"inflation",P_adj,N_adj,min_count=4) #P108, N125, S2002
gen_SO_df(sentence_sel_credit,"credit",P_adj,N_adj,min_count=4) #P47, N44, S1139
gen_SO_df(sentence_sel_MS,"MS",P_adj,N_adj,min_count=4) #P15, N27, S377
gen_SO_df(sentence_sel_MP,"MP",P_adj,N_adj,min_count=4) #P50, N58, S1596
gen_SO_df(sentence_sel_int,"int",P_adj,N_adj,min_count=4) #P112, N130, S3189
gen_SO_df(sentence_sel_liq,"liq",P_adj,N_adj,min_count=4) #P46, N82, S1693

SO_df_output = pd.read_csv("out\\results\\SO_df_output.csv",index_col=0)
SO_df_inflation = pd.read_csv("out\\results\\SO_df_inflation.csv",index_col=0)
SO_df_credit = pd.read_csv("out\\results\\SO_df_credit.csv",index_col=0)
SO_df_MP = pd.read_csv("out\\results\\SO_df_MP.csv",index_col=0)
SO_df_int = pd.read_csv("out\\results\\SO_df_int.csv",index_col=0)

SO_df_output["quadrant"] = SO_df_output.apply(lambda x:label_quadrant(x),axis=1)
SO_df_inflation["quadrant"] = SO_df_inflation.apply(lambda x:label_quadrant(x),axis=1)
SO_df_credit["quadrant"] = SO_df_credit.apply(lambda x:label_quadrant(x),axis=1)
SO_df_MP["quadrant"] = SO_df_MP.apply(lambda x:label_quadrant(x),axis=1)
SO_df_int["quadrant"] = SO_df_int.apply(lambda x:label_quadrant(x),axis=1)

SO_df_output["label_word"] = SO_df_output["word"].apply(lambda x:label_word(x))
SO_df_inflation["label_word"] = SO_df_inflation["word"].apply(lambda x:label_word(x))
SO_df_credit["label_word"] = SO_df_credit["word"].apply(lambda x:label_word(x))
SO_df_MP["label_word"] = SO_df_MP["word"].apply(lambda x:label_word(x))
SO_df_int["label_word"] = SO_df_int["word"].apply(lambda x:label_word(x))
SO_df_output.to_csv("out\\results\\SO_df_output.csv",encoding='utf-8-sig')
SO_df_inflation.to_csv("out\\results\\SO_df_inflation.csv",encoding='utf-8-sig')
SO_df_credit.to_csv("out\\results\\SO_df_credit.csv",encoding='utf-8-sig')
SO_df_MP.to_csv("out\\results\\SO_df_MP.csv",encoding='utf-8-sig')
SO_df_int.to_csv("out\\results\\SO_df_int.csv",encoding='utf-8-sig')
### calculate SO, by categories
output_pos = [line.strip() for line in open(r'dict\alternative_dict\output_pos.txt', 'r', encoding='utf-8-sig').readlines()]
output_neg = [line.strip() for line in open(r'dict\alternative_dict\output_neg.txt', 'r', encoding='utf-8-sig').readlines()]
inflation_pos = [line.strip() for line in open(r'dict\alternative_dict\inflation_pos.txt', 'r', encoding='utf-8-sig').readlines()]
inflation_neg = [line.strip() for line in open(r'dict\alternative_dict\inflation_neg.txt', 'r', encoding='utf-8-sig').readlines()]
credit_pos = [line.strip() for line in open(r'dict\alternative_dict\credit_pos.txt', 'r', encoding='utf-8-sig').readlines()]
credit_neg = [line.strip() for line in open(r'dict\alternative_dict\credit_neg.txt', 'r', encoding='utf-8-sig').readlines()]
mp_pos = [line.strip() for line in open(r'dict\alternative_dict\mp_pos.txt', 'r', encoding='utf-8-sig').readlines()]
mp_neg = [line.strip() for line in open(r'dict\alternative_dict\mp_neg.txt', 'r', encoding='utf-8-sig').readlines()]
interest_pos = [line.strip() for line in open(r'dict\alternative_dict\interest_pos.txt', 'r', encoding='utf-8-sig').readlines()]
interest_neg = [line.strip() for line in open(r'dict\alternative_dict\interest_neg.txt', 'r', encoding='utf-8-sig').readlines()]
sentence_sel_output["output"] = sentence_sel_output["ngram"].apply(lambda x:count_exp(x,output_pos,output_neg))
sentence_sel_inflation["inflation"] = sentence_sel_inflation["ngram"].apply(lambda x:count_exp(x,inflation_pos,inflation_neg))
sentence_sel_credit["credit"] = sentence_sel_credit["ngram"].apply(lambda x:count_exp(x,credit_pos,credit_neg))
sentence_sel_int["interest"] = sentence_sel_int["ngram"].apply(lambda x:count_exp(x,interest_pos,interest_neg))
sentence_sel_MP["mp"] = sentence_sel_MP["ngram"].apply(lambda x:count_exp(x,mp_pos,mp_neg))
MPR_result["n_output"] = zscore(sentence_sel_output.groupby("date")["output"].mean().values)
MPR_result["n_inflation"] = zscore(sentence_sel_inflation.groupby("date")["inflation"].mean().values)
MPR_result["n_credit"] = zscore(sentence_sel_credit.groupby("date")["credit"].mean().values)
MPR_result["n_int"] = zscore(sentence_sel_int.groupby("date")["interest"].mean().values)
MPR_result["n_mp"] = zscore(sentence_sel_MP.groupby("date")["mp"].mean().values)
### load weight(dict)
weight_LM = pkl.load(open("data\\weight_LM.pkl","rb"))
sentence_df_domestic_sel["ChFin"] = sentence_df_domestic_sel.apply(lambda x:calc_SO(x,weight_LM),axis=1)
for i in ["output","inflation","credit", "interest","MP"]:
    label = "ngram_" + i
    sentence_df_domestic_sel[label] = sentence_df_domestic_sel[i]*sentence_df_domestic_sel["ChFin"]
    MPR_result[label] = zscore(sentence_df_domestic_sel.groupby("date")[label].mean().values)


sentence_df_all.to_csv("out\\results\\sentence_df_all.csv",encoding='utf-8-sig')
sentence_df_domestic.to_csv("out\\results\\sentence_df_domestic.csv",encoding='utf-8-sig')
sentence_df_domestic_sel.to_csv("out\\results\\sentence_df_domestic_sel.csv",encoding='utf-8-sig')
pkl.dump(sentence_df_all,open("out\\results\\sentence_df_all.pkl","wb"))
pkl.dump(sentence_df_domestic,open("out\\results\\sentence_df_domestic.pkl","wb"))
pkl.dump(sentence_df_domestic_sel,open("out\\results\\sentence_df_domestic_sel.pkl","wb"))


sentence_sel_MP["credit"]=sentence_sel_MP["credit"]+sentence_sel_MP["MS"]+sentence_sel_MP["liquidity"]
sentence_sel_MP["credit"]=sentence_sel_MP["credit"].apply(lambda x:1 if x>0 else 0)
sentence_sel_MP.drop(labels=['MS','liquidity','stock'],axis=1,inplace=True)



MPR_result.to_csv("out\\results\\MPR_result.csv",encoding='utf-8-sig')
pkl.dump(MPR_result,open("out\\results\\MPR_result.pkl","wb"))

MPR_result[['ChFin', 'pbc', 'ChFin_pbc', 'ChFin_pbc_adj','GDP_g', 'M2_g', 'i_rate', 'CPI',
           'PPI','confidence']].corr().to_csv("out\\results\\corr.csv",encoding='utf-8-sig')

# plot
""" wordcloud
dict_pos = {}
dict_neg = {}
for k,v in dict_dtm.items():
    if k in P:
        dict_pos[k]=v
    if k in N:
        dict_neg[k]=v
count_P = 0
count_N = 0
for k,v in dict_pos.items():
    if v>=5:
        count_P+=1
for k,v in dict_neg.items():
    if v>=5:
        count_N+=1
dict_pos_sorted= sorted(dict_pos.items(), key=lambda d:d[1],reverse=True)
dict_neg_sorted= sorted(dict_neg.items(), key=lambda d:d[1],reverse=True)
pd.DataFrame(dict_pos_sorted).to_csv("out\\results\\LM_pos_count.csv",encoding="utf-8-sig")
pd.DataFrame(dict_neg_sorted).to_csv("out\\results\\LM_neg_count.csv",encoding="utf-8-sig")
from wordcloud import WordCloud
wc_neg = WordCloud(background_color='white',colormap="Greens_r",font_path='simfang.ttf',scale=20)
wc_neg.generate_from_frequencies(dict_neg)
plt.axis('off')
plt.grid('off')
wc_neg.to_file("out\\figs\\wc_neg.png")
plt.imshow(wc_neg)
wc_pos = WordCloud(background_color='white',colormap="Reds_r",font_path='simfang.ttf',scale=20)
wc_pos.generate_from_frequencies(dict_pos)
plt.axis('off')
plt.grid('off')
wc_pos.to_file("out\\figs\\wc_pos.png")
plt.imshow(wc_pos)
"""

sent_arrs = [['ChFin_overall', 'k', '-', 'o', "ChFin"],
['n_output', 'g', '--', 's', "ngram_output"],
['n_inflation', 'r', '--', '^', "ngram_inflation"],
['n_int', 'b', '-.', '*', "ngram_interest"]]
plot_multi_arrays(MPR_result, sent_arrs, "ngram")


sent_arrs = [['ChFin_overall', 'k', '-', 'o', "ChFin",1],  ['n_int', 'r', '--', 's', "ngram_interest",1],
             ['i_rate', 'b', '-.', '.', "利率",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "ngram_interest2", '银行间债券质押式回购平均利率')

sent_arrs = [['ChFin_overall', 'k', '-', 'o', "ChFin",1],  ['n_output', 'r', '--', 's', "ngram_output",1],
             ['GDP_g', 'b', '-.', '.', "GDP增速",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "ngram_output2", 'GDP增长指数')

sent_arrs = [['ChFin_overall', 'k', '-', 'o', "ChFin",1],  ['n_credit', 'r', '--', 's', "ngram_credit",1],
             ['M2_g', 'b', '-.', '.', "M2增速",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "ngram_credit2", 'M2增速')

sent_arrs = [['ChFin_overall', 'k', '-', 'o', "ChFin",1],  ['n_mp', 'r', '--', 's', "ngram_mp",1],
             ['M2_g', 'b', '-.', '.', "M2增速",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "ngram_mp2", 'M2增速')

sent_arrs = [['ChFin_overall', 'k', '-', 'o', "ChFin",1],  ['n_inflation', 'r', '--', 's', "ngram_inflation",1],
             ['CPI', 'b', '-.', '.', "CPI",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "ngram_inflation2", '居民消费价格指数')

sent_arrs = [['ChFin_overall', 'k', '-', 'o', "ChFin",1],  ['n_inflation', 'r', '--', 's', "ngram_inflation",1],
             ['PPI', 'b', '-.', '.', "PPI",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "ngram_inflation3", '工业生产者出厂价格指数')

sent_arrs = [['LM_all', 'k', '--', 'o', "LM",1],  ['int_rate', 'r', '-', 's', "int. re",1],
             ['GDP_g', 'b', '-.', '.', "GDP growth",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "LM&int_re&GDP_g", 'GDP growth', 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

sent_arrs = [['LM_all', 'k', '--', 'o', "LM",1],  ['int_rate', 'r', '-', 's', "int. re",1],
             ['M2_g', 'b', '-.', '.', "M2 growth",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "LM&int_re&M2_g", 'M2 growth', 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

sent_arrs = [['LM_all', 'k', '--', 'o', "LM",1], ['int_rate', 'r', '-', 's', "int. re",1],
             ['CPI', 'b', '-.', '.', "CPI",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "LM&int_re&CPI", 'CPI', 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

sent_arrs = [['LM_all', 'k', '--', 'o', "LM",1],
                ['int_rate', 'r', '-', 's', "int. re",1],
             ['PPI', 'b', '-.', '.', "PPI",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "LM&int_re&PPI", 'PPI', 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

sent_arrs = [['LM_all', 'k', '--', 'o', "LM",1],
                ['int_rate', 'r', '-', 's', "int. re",1],
             ['confidence', 'b', '-.', '.', "confidence",2]]
plot_multi_arrays2(MPR_result,sent_arrs, "LM&int_re&conf", 'confidence', 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')


"""
sns.scatterplot(data=SO_df_ngram[SO_df_ngram['seed_word']!=0], x='SO_cos', y='SO_PPMI', hue='isin_lexicon', hue_order=['正向词','负向词'],
                    alpha=0.5,palette={'正向词':'r',"负向词":'g'})
plt.xlim(-3, 3)
plt.ylim(-6, 6)
plt.savefig('out\\figs\\ChFin_SO.jpg', dpi=600)
plt.show()

sns.boxplot(x='isin_lexicon', y='SO_cos', data=SO_df_ngram)
plt.xticks(fontsize=14)
plt.ylabel("SO_cos",fontsize=14)
plt.ylim(-4,4)
plt.savefig("out\\figs\\SO_df_ngram_SO_cos.png",dpi=600)


sns.jointplot(x='SO_cos', y='SO_PPMI', data=SO_df_ngram,hue='isin_lexicon', hue_order=['正向词','负向词'],
              xlim=[-4,4], ylim=[-8,8],kind="scatter", palette={'正向词':"#B34C6E","负向词":"#4CB391"})
"""
###fig 5 ChFin: Semantic Orientation
sns.jointplot(x='SO_cos', y='SO_PPMI', data=SO_df_ngram[SO_df_ngram['seed_word']==-1],
              xlim=[-4,4], ylim=[-8,8], kind="hex", color="#4CB391")
plt.vlines(0, -8, 8, colors='#4CB391', linestyles=':')
plt.hlines(0, -4, 4, colors='#4CB391', linestyles=':')
plt.annotate(s="ChFin负向词",xy=(2,6),xytext=(2,6),color="#4CB391",fontsize=15)
plt.savefig("out\\figs\\ChFin_neg_SO.png",dpi=600)
sns.jointplot(x='SO_cos', y='SO_PPMI', data=SO_df_ngram[SO_df_ngram['seed_word']==1],
              xlim=[-4,4], ylim=[-8,8], kind="hex", color="#B34C6E")
plt.vlines(0, -8, 8, colors='#B34C6E', linestyles=':')
plt.hlines(0, -4, 4, colors='#B34C6E', linestyles=':')
plt.annotate(s="ChFin正向词",xy=(2,6),xytext=(2,6),color="#B34C6E",fontsize=15)
plt.savefig("out\\figs\\ChFin_pos_SO.png",dpi=600)

pkl.dump(doc_LM_df, open("out\\results\\doc_LM_df.pkl", "wb"))
doc_LM_df.to_csv("out\\results\\doc_LM_df.csv", encoding='utf-8-sig')
pkl.dump(sentence_df, open("out\\results\\sentence_df.pkl", "wb"))
sentence_df.to_csv("out\\results\\sentence_df.csv", encoding='utf-8-sig')



P+N-weight_INF_exp.keys()
pkl.dump(weight_INF_exp,open("data\\weight_INF_exp.pkl","wb"))
weight_INF_combine = {**weight_INF_exp,**weight_INF}
pkl.dump(weight_INF_combine,open("data\\weight_INF_combine.pkl","wb"))

sent_arrs = [['LM_t', 'k', '--', 'o', "LM"],
                ['INF', 'g', '-', 's', "INF"],
                 ['INF_combine', 'r', '-', 's', "INF combined"]]
plot_multi_arrays(doc_LM_df, sent_arrs, "INF_2_0306", 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

sent_arrs = [['LM_t', 'k', '--', 'o', "LM"],
                ['LM_exp_d_t', 'r', '-', 's', "LM new"],]
plot_multi_arrays(doc_LM_df, sent_arrs, "LM_0306", 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')







