import pandas as pd
import re
import jieba
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
#from ngram import gen_ngram
import time
import seaborn as sns
import datetime
import glob

"""
PENG Limin, 3rd Mar. 2022
"""
"""
os.chdir('D:\\PLM\\宏观金融seminar\\Seminar2021-2022\\央行沟通词典\\code')
sns.set_theme(color_codes=True)
sns.set_palette("muted")
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
### load user_dict
user_dicts = os.listdir("dict")
user_dicts = [d for d in user_dicts if ".txt" in d]
for i in user_dicts:
    path = "dict\\" + i
    jieba.load_userdict(path)


degree = [line.strip() for line in open(r'dict\degreeDictfull.txt', 'r', encoding='utf-8-sig').readlines()]
degree_dict = {}
for i in degree:
    item = i.split(',')
    try:
        degree_dict[item[0]] = float(item[1])
    except:
        print(i)

# global settings
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
os.chdir('D:\\PLM\\宏观金融seminar\\Seminar2021-2022\\央行沟通词典\\code')

# load user defined dictionaries as word lists, globally
stopwords = [line.strip() for line in open(r'dict\百度停用词表中文.txt', 'r', encoding='utf-8-sig').readlines()]  # 停用词词典
Pwords = [line.strip() for line in open(r'dict\LM_pos.txt', 'r', encoding='utf-8-sig').readlines()]
Nwords = [line.strip() for line in open(r'dict\LM_neg.txt', 'r', encoding='utf-8-sig').readlines()]
Pwords2 = [line.strip() for line in open(r'dict\INF_pos.txt', 'r', encoding='utf-8-sig').readlines()] # alternative lexicon
Nwords2 = [line.strip() for line in open(r'dict\INF_neg.txt', 'r', encoding='utf-8-sig').readlines()] # alternative lexicon
degreeDict = [line.strip() for line in open(r'dict\degreeDict.txt', 'r', encoding='utf-8-sig').readlines()] #
notDict = [line.strip() for line in open(r'dict\notDict.txt', 'r', encoding='utf-8-sig').readlines()]
stopwords = list(set(stopwords)-set(Pwords)-set(Nwords)-set(degreeDict)-set(notDict)-set(Pwords2)-set(Nwords2)) # 去重后的停用词

"""

def read_txt(filename,df_name):
    """
    read txt files into a pandas dataframe
    """
    files = glob.glob('data\\{}\\*.txt'.format(filename))
    quarter_list = []
    date_list = []
    text_list = []
    for filename in files:
        quarter = '20'+filename[-8:-6]+'q'+filename[-5]
        date = quarter[:4]
        if int(quarter[5]) == 1:
            date +=  '0331'
        if int(quarter[5]) == 2:
            date += '0630'
        if int(quarter[5]) == 3:
            date += '0930'
        if int(quarter[5]) == 4:
            date += '1231'
        with open(filename,encoding='utf-8-sig') as f:
            text = f.readlines()
            text = "".join(text)
            text = ''.join(''.join(text.split('\n')).split('\t'))
            text = text.replace(" ", "")  # 去掉空格
            print(quarter)
            print(date)
            print(text)
            f.close()
        quarter_list.append(quarter)
        date_list.append(date)
        text_list.append(text)

    dt = pd.DataFrame({"date":date_list,"text":text_list,"quarter":quarter_list})
    dt.date = pd.to_datetime(dt.date,format="%Y/%m/%d")
    dt.to_csv("data\\{}.csv".format(df_name),encoding='utf-8-sig')
    pkl.dump(dt,open("data\\{}.pkl".format(df_name),"wb"))

def zscore(arr):
    """
    normalized values of the given array
    :param arr: array
    :return: normalized values of the given array
    """
    mu = arr.mean()
    sigma = arr.std()
    return (arr - mu) / sigma

def build_dict(word_list,filename):
    """
    build dict (.txt) from list
    :param word_list: list of words
    :param filename: filename to save
    :return: save dict(dict\\filename.txt)
    """
    print("building dict{}".format(filename+'.txt'))
    print("dict size:{} words".format(len(word_list)))
    with open("dict\\{}.txt".format(filename), "a",encoding='utf-8-sig') as f:
        for line in word_list:
            f.write(str(line) + '\n')
    f.close()

def build_weight_dict(pos,neg,filename):
    weight_dict = {}
    num_p = 0
    num_n = 0
    for p in pos:
        weight_dict[p] = 1
        num_p += 1
    for n in neg:
        weight_dict[n] = -1
        num_n += 1
    print("num of pos words: ",num_p,"\nnum of neg words: ",num_n)
    pkl.dump(weight_dict,open("data\\{}.pkl".format(filename),"wb"))
    return weight_dict

def calc_SO(row,weight_dict,rules=True):
    """
    Hutto & Gilbert (2014)
    scoring sentences based on lexicons, and the corresponding weight (weight_k)
    can only apply to the sentence_df
    :param row: a row in the sentence_df
    :param rules: whether to use semantic rules, default true, boolean
    :param weight: the corresponding weight of words in the lexicon , dict
    :param pos_threshold: threshold above which a sentence is classified to be positive ,float
    :param neg_threshold: threshold above which a sentence is classified to be negative,float
    :return: SO (semantic orientation of words in the corpus), dict
    """

    sent_keys = list(weight_dict.keys())
    word_list = row['ngram'] # list of word

    # score sentences
    sent_list = [w for w in word_list if w in sent_keys]
    sent_ind = [i for i in range(len(word_list)) if word_list[i] in sent_list]
    num_sent = len(sent_ind) # number of sentiment words
    sent_score = 0  # sentiment score of the sentence
    if num_sent > 0: # one or more than one sentiment words, calculate sentence sentiment score by its sentimental units
        for i in range(num_sent):
            sent_word = sent_list[i] # the corresponding sentiment word
            sent_class = weight_dict[sent_word]
            print("sent_word: ",sent_word, "weight: ",sent_class)
            if i==0:
                begin_ind = 0 # begin index of the sentiment unit
            else:
                begin_ind = sent_ind[i-1]+1
            end_ind = sent_ind[i] # end index of the sentiment unit in word_list
            print("begin_ind: ", begin_ind, "end_ind: ", end_ind)
            intensity = 1
            negation = 1
            if rules:
            # sentiment unit
                for j in range(begin_ind,end_ind):
                    # negation rule
                    w = word_list[j]
                    w_p1 = word_list[j+1]
                    if ((w in notDict) and (not ((w=="没有")&(w_p1=="改变")or(w=="不会")&(w_p1=="改变")))) \
                            or (w=="失业率") or (w=="就业压力") or (w=="不良贷款"):
                        negation*=(-1)
                        print(w, " in notDict", " negation = ", negation)
                    # intensity rule
                    """
                    if (w in degree_dict) and (w not in notDict):
                        intensity*=degree_dict[w]
                        print(w, " in degree_dict", degree_dict[w], " intensity = ", intensity)
                    """
            unit_score = intensity*negation*sent_class # sentiment score of the unit
            print("unit: ",word_list[begin_ind:end_ind],"sentiment word: ",word_list[end_ind])
            print("unit_score: ",unit_score)
            sent_score += unit_score
    else: # no sentiment word
        sent_score = 0
    sent_score = sent_score/row["num_word"]
    return sent_score

def re_match(doc,pattern1,pattern2='',delete_external=True):
    externalDict = [line.strip() for line in open(r'dict\alternative_dict\externalDict.txt', 'r',encoding='utf-8-sig').readlines()]
    sentences = re.split(r'(\!|\?|。|！|？|，|,)', doc)
    sentences = [s for s in sentences if len(s) > 1]
    if delete_external:
        external_sentences = []
        for s in sentences:
            for ext in externalDict:
                if ext in s:
                    external_sentences.append(s)
                    break
        sentences = [s for s in sentences if s not in external_sentences]
    pattern1_result = []
    pattern2_result = []
    for s in sentences:
        result = re.findall(pattern1, s)
        if len(result) > 0:
            #print(result)
            pattern1_result.append(result)
    if pattern2 != '':
        for s in sentences:
            result = re.findall(pattern2, s)
            if len(result) > 0:
                #print(result)
                pattern2_result.append(result)
    count_diff = len(pattern1_result)-len(pattern2_result)
    return pattern1_result,pattern2_result,count_diff

def plot_multi_arrays(dt,sent_arrs,filename,window=0):
    """
    plot multiple arrays on the same x_axis
    :param dt: dataframe with a time_index (dt.date), dataframe
    :param sent_arrs: [varname:string, color:string, linestyle:string, marker:string,label:string]
    :param filename: path to save the plot
    :param window: window length of moving average, default 0 (no ma)
    :return: saved plot
    """
    plt.figure(figsize=(16, 9))
    num = len(sent_arrs)
    df = dt.copy()
    #df = df[df["type"]=="货币政策报告"]
    ts = df.quarter
    for i in range(num):
        if window == 0:
            arr = df[sent_arrs[i][0]]
        else:
            arr = df.rolling(window=window)[sent_arrs[i][0]].mean()
        color = sent_arrs[i][1]
        linestyle = sent_arrs[i][2]
        marker = sent_arrs[i][3]
        label = sent_arrs[i][4]
        plt.plot(ts,arr,color,linestyle=linestyle, marker=marker,label=label)
    #x_labels = list(pd.date_range(begin_t, end_t, freq=freq).to_period('Q').strftime("%Yq%q"))
    plt.xticks(range(0,len(ts),2),rotation=90,fontsize=15)
    #plt.xlabel("date",fontsize=15)
    #plt.ylabel("orientation",fontsize=15)
    plt.ylim([-4, 4])
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15,loc="upper right")
    plt.savefig("out\\figs\\{}.png".format(filename), format='png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_multi_arrays2(dt,sent_arrs,filename,ax2ylabel,window=0):
    """
    plot multiple arrays on the same x_axis
    :param dt: dataframe with a time_index (dt.date), dataframe
    :param sent_arrs: [varname:string, color:string, linestyle:string, marker:string,label:string,axis:1(wv) or 2(PMI)]
    :param filename: path to save the plot
    :param ax2ylabel: ax2ylabel (label of the 2nd y axis)
    :param window: window length of moving average, default 0 (no ma)
    :return: saved plot
    """
    fig,ax1=plt.subplots(figsize=(16, 9))
    ax2 = ax1.twinx()
    num = len(sent_arrs)
    df = dt.copy()
    ts = df.quarter
    for i in range(num):
        if window == 0:
            arr = df[sent_arrs[i][0]]
        else:
            arr = df.rolling(window=window)[sent_arrs[i][0]].mean()
        color = sent_arrs[i][1]
        linestyle = sent_arrs[i][2]
        marker = sent_arrs[i][3]
        label = sent_arrs[i][4]
        if sent_arrs[i][5]==1:
            ax1.plot(ts,arr,color,linestyle=linestyle, marker=marker,label=label)
        else:
            ax2.plot(ts, arr, color, linestyle=linestyle, marker=marker, label=label)

    fig.legend(loc="upper right",bbox_to_anchor=(1,1),bbox_transform=ax1.transAxes,fontsize=15)
    #x_labels = list(pd.date_range(begin_t, end_t, freq=freq).to_period('Q').strftime("%Yq%q"))
    plt.xticks(range(0,len(ts),2),rotation=90,fontsize=15)

    #ax1.set_xlabel("date", fontsize=14)
    ax1.set_ylabel("文本倾向", fontsize=15)
    ax1.set_ylim((-4,4))
    ax2.set_ylabel(ax2ylabel, fontsize=15)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    #plt.legend(fontsize=14)
    for label in ax1.get_xticklabels():
        label.set_rotation(90)
    plt.savefig("out\\figs\\{}.png".format(filename), format='png', dpi=300, bbox_inches='tight')
    plt.show()

def label_sentence(s):
    external = [line.strip() for line in open(r'dict\alternative_dict\externalDict.txt', 'r', encoding='utf-8-sig').readlines()]
    delete = ["推动", "推进", "促进", "有利于", "有助于", "培养", "引导", "保证", "确保", "调整", "转变", "改善",
              "深化", "健全", "支持", "鼓励", "保持", "坚持","激发"]
    inflation = ["通胀","通货膨胀","通缩","通货紧缩","消费价格","消费品价格","食品价格","粮食价格","农产品价格","猪肉价格",
                 "商品价格","用品价格", "服务价格", "生产资料价格","生产价格","生产者价格","原材料价格","要素价格",
                 "工业品价格","出厂价格","购进价格","价格水平","价格总水平","价格指数","价格走势","价格涨幅","价格上涨幅度",
                 "价格涨势","价格变化","价格指标","物价", "CPI","cpi","居民消费价格","PPI","ppi","工业生产者出厂价格"]
    credit = ["货币信贷","社会融资规模","银行信贷","银行贷款","金融机构贷款","金融机构人民币贷款"]
    MS = ["货币供应","货币供给","流通中现金","M0","m0","狭义货币","M1","m1","广义货币","M2","m2"]
    output = ["中国经济","我国经济","国内经济","国民经济","当前经济","宏观经济","经济增长","经济前景","经济增速","经济发展",
              "经济运行","经济环境", "国内生产总值","GDP","gdp","总需求","总供给","总消费","总产出","最终需求",
                              "国内需求","消费","内需","内外需",
              "收入","工资水平","投资","出口","工业生产","工业增加值","工业完成值","工业增速","景气","失业率","就业"]
    stock = ["股市","股票市场","二级市场","沪市","深市","A股","沪深","股价","股票价格","行情","市场信心", "股指",
             "股票指数","上证","深证","沪指","深指"]
    int_r = ["利率","收益率","准备金率", "贴现率", "加息", "升息",  "降息"]

    label=''
    for external_w in external:
        if external_w in s:
            label += " external"
            break
    for inflation_w in inflation:
        if inflation_w in s:
            label += " inflation"
            break
    for output_w in output:
        if output_w in s:
            label += " output"
            break
    for credit_w in credit:
        if credit_w in s:
            label += " credit"
            break
    for MS_w in MS:
        if MS_w in s:
            label += " MS"
            break
    if ("货币政策" in s) or ("利率政策" in s):
        label += " MP"
    if "流动性" in s:
        label += "流动性"
    for int_r_w in int_r:
        if int_r_w in s:
            label += " int_r"
            break
    for stock_w in stock:
        if stock_w in s:
            label += " stock"
            break
    for del_w in delete:
        if del_w in s:
            label += " delete"
            break
    return label

def count_label(label):
    output = 0
    credit = 0
    MS = 0
    liquidity = 0
    inflation=0
    MP=0
    int_r=0
    stock=0
    if label != "":
        if "external" not in label:
            if ("output" in label):
                output = 1
            if ("inflation" in label):
                inflation=1
            if ("credit" in label):
                credit=1
            if ("MS" in label):
                MS=1
            if ("流动性" in label):
                liquidity=1
            if ("int_r" in label):
                int_r=1
            if ("MP" in label):
                MP=1
            if ("stock" in label):
                stock=1
    return (output,inflation,credit,MS,liquidity,int_r,MP,stock)

"""
sentence_df_domestic_sel["label2"]=sentence_df_domestic_sel["label"].\
    apply(lambda x:count_label(x))
sentence_df_domestic_sel[["output","inflation","credit","MS","liquidity","int_r","MP","stock"]]=sentence_df_domestic_sel.\
    apply(lambda x:x["label2"],axis=1,result_type='expand')
sentence_df_domestic_sel.groupby('date')[["output","inflation","credit","MS","liquidity","int_r","MP","stock"]].sum().plot()
"""

def MP_stance(dtime_):
    """
    monetary policy stance, classified by Lin & Zhao (2015), from 2001Q1 to 2013Q4
    :param dtime_: datetime object (or string)
    :return:
    """
    stance=""
    dtime = dtime_.strftime("%Y/%m/%d") # if datetime object
    if int(dtime[:4])<2014:
        stance="neutral"
        if dtime in  ["2001/12/31","2002/3/31","2005/3/31","2008/9/30","2008/12/31","2009/3/31","2009/6/30","2009/12/31",
                      "2011/12/31","2012/3/31","2012/6/30","2012/9/30"]:
            stance="dovish"
        if dtime in ["2003/12/31","2004/3/31","2004/6/30","2004/9/30","2005/12/31","2006/3/31","2006/6/30","2006/9/30","2006/12/31",
                        "2007/3/31","2007/6/30","2007/9/30","2007/12/31","2008/3/31","2008/6/30","2010/3/31","2010/6/30",
                        "2010/9/30","2010/12/31","2011/3/31","2011/6/30", "2011/9/30"]:
            stance="hawkish"
    return stance

def label_quadrant(row):
    quadrant = 0
    if (row["SO_cos"]>0) and (row["SO_PPMI"]>0):
        quadrant = 1
    if (row["SO_cos"] < 0) and (row["SO_PPMI"] > 0):
        quadrant = 2
    if (row["SO_cos"] < 0) and (row["SO_PPMI"] < 0):
        quadrant = 3
    if (row["SO_cos"] > 0) and (row["SO_PPMI"] < 0):
        quadrant = 4
    return quadrant
"""
SO_df_output["quadrant"] = SO_df_output.apply(lambda x:label_quadrant(x),axis=1)
"""

def label_word(x):
    label=""
    if x in pbc_pos:
        label = "pbc_pos"
    if x in pbc_neg:
        label = "pbc_neg"
    if x in pbc_del:
        label = "pbc_del"
    return label
"""
SO_df_output["label_word"] = SO_df_output["word"].apply(lambda x:label_word(x))
"""

def count_exp(x,P,N):
    """
    count negative and positive expressions in a PARSED list
    :param x: a PARSED list of strings, like [发挥, 发挥好, 发挥好市场利率定价, ...]
    :param P: list of positive expressions
    :param N: list of negative expressions
    :return: net count (i.e., P-N)
    """
    P_count = 0
    N_count = 0
    for i in x:
        if i in P:
            P_count +=1
        if i in N:
            N_count +=1
    net_count = P_count-N_count
    return net_count

if __name__ == '__main__':
    """
    weight_LM={}
    num_p=0
    num_n=0
    for i in Pwords_:
        weight_LM[i]=1
        num_p+=1
    for i in Nwords_:
        weight_LM[i]=-1
        num_n+=1 
    sentence_df = pkl.load(open("out\\results\\sentence_df.pkl", "rb"))
    doc_LM_df = pkl.load(open("out\\results\\doc_LM_df.pkl", "rb"))
    doc_LM_df = pd.read_csv("out\\results\\doc_LM_df.csv", index_col=0)

    weight_LM = pkl.load(open("data\\weight_LM.pkl", "rb"))
    weight_LM_cos = pkl.load(open("data\\weight_LM_cos.pkl", "rb"))
    weight_LM_PPMI = pkl.load(open("data\\weight_LM_PPMI.pkl", "rb"))
    weight_LM_exp_cos = pkl.load(open("data\\weight_LM_exp_cos.pkl", "rb"))
    weight_LM_exp_PPMI = pkl.load(open("data\\weight_LM_exp_PPMI.pkl", "rb"))

    weight_LM_cos_d = weight_LM_cos.copy() # 深拷贝
    for i in weight_LM_cos.keys():
        if i in weight_LM.keys():
            weight_LM_cos_d.pop(i)
    weight_LM_PPMI_d = weight_LM_PPMI.copy()  # 深拷贝
    for i in weight_LM_PPMI.keys():
        if i in weight_LM.keys():
            weight_LM_PPMI_d.pop(i)
    weight_LM_exp_cos_d = weight_LM_exp_cos.copy()  # 深拷贝
    for i in weight_LM_exp_cos.keys():
        if i in weight_LM.keys():
            weight_LM_exp_cos_d.pop(i)
    weight_LM_exp_PPMI_d = weight_LM_exp_PPMI.copy()  # 深拷贝
    for i in weight_LM_exp_PPMI.keys():
        if i in weight_LM.keys():
            weight_LM_exp_PPMI_d.pop(i)
    pkl.dump(weight_LM_cos_d,open("data\\weight_LM_cos_d.pkl","wb"))
    pkl.dump(weight_LM_PPMI_d, open("data\\weight_LM_PPMI_d.pkl", "wb"))
    pkl.dump(weight_LM_exp_cos_d, open("data\\weight_LM_exp_cos_d.pkl", "wb"))
    pkl.dump(weight_LM_exp_PPMI_d, open("data\\weight_LM_exp_PPMI_d.pkl", "wb"))

    sentence_df["s_LM"] = sentence_df.apply(lambda x:calc_SO(x, weight_LM, rules=True),axis=1)
    sentence_df["s_LM_cos"] = sentence_df.apply(lambda x: calc_SO(x,  weight_LM_cos, rules=True), axis=1)
    sentence_df["s_LM_PPMI"] = sentence_df.apply(lambda x: calc_SO(x, weight_LM_PPMI, rules=True), axis=1)
    sentence_df["s_LM_exp_cos"] = sentence_df.apply(lambda x: calc_SO(x, weight_LM_exp_cos, rules=True), axis=1)
    sentence_df["s_LM_exp_PPMI"] = sentence_df.apply(lambda x: calc_SO(x, weight_LM_exp_PPMI, rules=True), axis=1)
    sentence_df["s_LM_cos_d"] = sentence_df.apply(lambda x: calc_SO(x, weight_LM_cos_d, rules=True), axis=1)
    sentence_df["s_LM_PPMI_d"] = sentence_df.apply(lambda x: calc_SO(x, weight_LM_PPMI_d, rules=True), axis=1)
    sentence_df["s_LM_exp_cos_d"] = sentence_df.apply(lambda x: calc_SO(x, weight_LM_exp_cos_d, rules=True), axis=1)
    sentence_df["s_LM_exp_PPMI_d"] = sentence_df.apply(lambda x: calc_SO(x, weight_LM_exp_PPMI_d, rules=True), axis=1)

    doc_df = sentence_df.groupby("date").count().reset_index()[["date","sentence"]]
    doc_df.rename(columns={"sentence":"num_sentence"},inplace=True)

    doc_df2 = sentence_df.groupby("date").sum().reset_index()[["date", 's_LM', 's_LM_cos', 's_LM_PPMI', 's_LM_exp_cos',
       's_LM_exp_PPMI','s_LM_cos_d', 's_LM_PPMI_d', 's_LM_exp_cos_d','s_LM_exp_PPMI_d']]
    doc_LM_df = pd.merge(doc_df,doc_df2,how = 'inner', on = 'date')
    for i in ['s_LM', 's_LM_cos', 's_LM_PPMI', 's_LM_exp_cos','s_LM_exp_PPMI','s_LM_cos_d', 's_LM_PPMI_d', 's_LM_exp_cos_d','s_LM_exp_PPMI_d']:
        doc_LM_df[i] = doc_LM_df[i]/doc_LM_df['num_sentence']
        doc_LM_df[i] = zscore(doc_LM_df[i])
        new_name = i[2:]+'_t'
        doc_LM_df.rename(columns={i:new_name},inplace=True)
    doc_LM_df[['LM_t', 'LM_cos_t', 'LM_PPMI_t', 'LM_exp_cos_t','LM_exp_PPMI_t','LM_cos_d_t', 'LM_PPMI_d_t',
               'LM_exp_cos_d_t','LM_exp_PPMI_d_t']].corr().to_csv("out\\results\\corr_LM_0302.csv")

    sent_arrs = [['LM_t', 'g', '-', 'o', "LM"],
                 ['LM_PPMI_d_t', 'r', '-', 's', "LM_SO_PPMI1"],
                 ['LM_cos_d_t', 'b', '-', '^', "LM_SO_cos1"]]
    plot_multi_arrays(doc_LM_df, sent_arrs, "LM_PPMI_cos_diff0301", 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

    sent_arrs = [['LM_t', 'g', '-', 'o', "LM"],
                 ['LM_PPMI_d_t', 'r', '-', 's', "LM_SO_PPMI"]]
    plot_multi_arrays(doc_LM_df, sent_arrs, "LM_PPMI_diff0301", 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

    sent_arrs = [['LM_t', 'g', '-', 'o', "LM"],
                 ['LM_cos_d_t', 'b', '-', '^', "LM_SO_cos"]]
    plot_multi_arrays(doc_LM_df, sent_arrs, "LM_cos_diff0301", 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

    sent_arrs = [['LM_t', 'g', '-', 'o', "LM"],
                 ['LM_PPMI_d_t', 'r', '-', 's', "LM_SO_PPMI"],
                 ['LM_cos_d_t', 'b', '-', '^', "LM_SO_cos"]]
    plot_multi_arrays(doc_LM_df, sent_arrs, "LM_PPMI_cos_diff_2ma_0301", 3, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')

    sent_arrs = [['LM_t', 'g', '-', 'o', "LM"],
                 ['LM_exp_PPMI_d_t', 'r', '-', 's', "LM_PPMI"],
                 ['LM_exp_cos_d_t', 'b', '-', '^', "LM_cos"]]
    plot_multi_arrays(doc_LM_df, sent_arrs, "LM_PPMI_cos_diff_0302", 0, begin_t='2001q1', end_t='2021q1', freq='Q-DEC')
    doc_LM_df = pd.read_csv("out\\results\\doc_LM_df.csv",index_col=0)
    doc_LM_df.date = pd.to_datetime(doc_LM_df.date)
    pkl.dump(doc_LM_df, open("out\\results\\doc_LM_df.pkl", "wb"))
    doc_LM_df.to_csv("out\\results\\doc_LM_df.csv",encoding='utf-8-sig')
    pkl.dump(sentence_df, open("out\\results\\sentence_df.pkl", "wb"))
    sentence_df.to_csv("out\\results\\sentence_df.csv", encoding='utf-8-sig')

    ### alternative measures
    sentence_df["label"] = sentence_df["sentence"].apply(lambda x:label_sentence(x))
    sentence_df_sel = sentence_df[sentence_df["label"]!="external"]
    sentence_df_sel = sentence_df_sel[sentence_df_sel["label"]!=""]
    sentence_df_sel["stance"] = sentence_df_sel["date"].apply(lambda x:MP_stance(x))
    #sentence_df_sel.date = pd.to_datetime(sentence_df_sel.date)
    sentence_df_sel = sentence_df_sel.sort_values(by='date').reset_index().drop(columns="index")
    sentence_train_dov = sentence_df_sel[sentence_df_sel["stance"]=="dovish"]
    sentence_train_hawk = sentence_df_sel[sentence_df_sel["stance"]=="hawkish"]
    sentence_train_neutral = sentence_df_sel[sentence_df_sel["stance"] == "neutral"]
    sentence_train_dov = sentence_train_dov.reset_index().drop(columns="index")
    sentence_train_hawk = sentence_train_hawk.reset_index().drop(columns="index")
    sentence_train_neutral = sentence_train_neutral.reset_index().drop(columns="index")
    pkl.dump(sentence_train_dov,open("data\\sentence_train_dov.pkl","wb"))
    pkl.dump(sentence_train_hawk, open("data\\sentence_train_hawk.pkl", "wb"))
    pkl.dump(sentence_train_neutral, open("data\\sentence_train_neutral.pkl", "wb"))
    sentence_df_sel.to_csv("data\\sentence_df_sel.csv",encoding="utf-8-sig")
    pkl.dump(sentence_df_sel,open("data\\sentence_df_sel.pkl", "wb"))
    """
    t='调査失业率有所下降，就业形势总体稳定'
    label_sentence(t)
