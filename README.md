# Lexical Semantic Orientation
calculating lexical semantic orientation following Turney &amp; Littman (2003), using word2vec &amp; PPMI.

## 代码说明
> last update: 20th April, 2022
> author: PENG Limin

参考 Turney and Littman (2003) ，使用余弦相似度 (cosine similarity) 和正点交互熵 (positive pointwise mutual information, PPMI) 计算特定语域中全部词语的语义倾向 (semantic orientation)。

与 Turney and Littman (2003) 不同的是，余弦相似度使用的词向量由 word2vec 计算得来，而非 LSA (latent semantic analysis)。此外，使用了 PPMI，而非原文的PMI。

Turney and Littman (2003) 对语义倾向的抽象定义由下式给出：

$$
SO\_A (word) = \sum_{pword\in P}A(word,pword)-\sum_{nword\in N}A(word,nword) \tag{1}
$$

它衡量了特定词语 $word$ 与给定一正向词集合 $P$ 和负向词集合 $N$ 之间关联性的差异。其中 $pword$ 是正向词集合 $P$ 中某词语，$A(word,pword)$ 是 $word$ 和 $pword$ 之间的关联性函数。$SO\_A (word)$ 是根据正向词集合 $P$ 和负向词集合 $N$ 判定的词语语义倾向，当 $SO\_A (word)$ 取值相对较大且为正时，意味着词语 $word$ 很可能具有正向含义，反之则可能有负向含义。

在 $(1)$ 式基础上，以正点交互熵 $PPMI$ 和余弦相似度 $cos$ 作为关联性函数 $A$，具体定义以下两种语义倾向函数：

$$
SO\_PPMI (word) = \sum_{pword\in P}PPMI(word,pword)-\sum_{nword\in N}PPMI(word,nword) \tag{2}
$$

$$
SO\_cos (word) = \sum_{pword\in P}cos(word,pword)-\sum_{nword\in N}cos(word,nword) \tag{3}
$$

$(2)$ 式中的正点交互熵 $PPMI$ 定义如下，$P(word_i,word_j)$ 是 $word_i$ 和 $word_j$ 的共同出现频率：

$$
PMI(word_i,word_j) = log_2\frac{P(word_i,word_j)}{P(word_i)P(word_j)}, P(word_i,word_j)\neq 0 
$$

$$
PPMI(word_i,word_j) = max(PMI(word_i,word_j),0) \tag{4}
$$

$word_i$ 和 $word_j$ 的余弦相似度定义为词向量内积与模的比值:
$$
cos(word_i,word_j) = \frac{\vec{v}\cdot \vec{w}}{|\vec{v}||\vec{w}|} = \frac{\sum_{k=1}^{K}v_k w_k}{\sqrt{\sum_{k=1}^{K}v_k^2}\sqrt{\sum_{k=1}^{K}w_k^2}} \tag{5}
$$

word2vec 模型将每个词语表示为一个向量，$\vec{v}=(v_1,v_2,...,v_k)$ 和 $\vec{w}=(w_1,w_2,...,w_k)$ 分别是 $word_i$ 和 $word_j$ 在 $K$ 维向量空间中的坐标。

### 1. 可执行程序：`main.py`

cos_SO 和 PPMI_SO 计算函数：`gen_SO_df(df,filename,P,N,min_count=6) (line 53)`

### 2. 函数文件

#### 2.1. `PMI_SO.py`：计算 PPMI_SO

term-term co-occurence matrix 词语-词语共现矩阵计算函数: `build_ttm(sentence,posword,negword,vocab,ttm) (line 131)`

PPMI_SO 计算函数: `calc_PPMI_SO(ttm,vocab,Pwords,Nwords) (line 199)`

#### 2.2. `w2v_SO.py`：训练 word2vec 模型，计算 cos_SO

cos_SO 计算函数: `calc_cosine_SO(model, Pwords, Nwords,vocab) (line 475)`

#### 2.3. `sentence_sent_score.py`：计算句子语调

句子语调计算函数: `calc_SO(row,weight_dict,rules=True) (line 66)`

### 文件夹

#### data: 存放数据

#### out: 存放结果
- figs: 存放图片
- results: 存放各种计算结果，主要是 pandas dataframe 和对应的 pkl 文件
- models: 存放训练过的 w2v 模型

#### dict: 存放各种词典
- 自定义词典
  - notDict.txt: 知网否定词词典
  - degreeDict.txt: 知网程度副词词典
  - stopwords.txt: 百度停用词词典
  - sougou_caijing: 搜狗财经词库
  - sougou_jinrong: 搜狗金融词库
  - pbc_add.txt: mpr 里一些特定表述
- 情感词典
