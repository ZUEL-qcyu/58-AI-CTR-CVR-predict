from gensim.models import Word2Vec
import torch.nn as nn
import pandas as pd
import tqdm
train = pd.read_feather('./data/train.feather')
test = pd.read_feather('./data/test.feather')

data = pd.concat([train[['user_profile', 'click_job_id','jobs_feature','job_content_label']],
                  test[['user_profile', 'click_job_id','jobs_feature','job_content_label']]], axis=0)
data = data.fillna('-99')
data = data.replace({'':'-99'})
for i in ['user_profile', 'click_job_id','jobs_feature','job_content_label']:
    sentence = []
    for line in list(data[i].values):
        sentence.append([str(l) for l in line.split(',')])
    print('training...{}'.format(i))
    if i in ['user_profile','job_content_label']:#短文本、强局部关联
        model = Word2Vec(sentence,window=2,# 捕捉紧密关联技能
            epochs=10,
            vector_size=16, # 用户画像维度无需过高
            min_count=1     # 过滤低频标签
        )
    elif i == 'click_job_id':#离散ID，需捕捉职位共现模式
        model = Word2Vec(sentence,window=5, # 扩大窗口捕捉用户浏览序列模式
            epochs=12,
            vector_size=24, # ID类特征需要更高维度编码
            min_count=1
        )
    else:
        model = Word2Vec(sentence,window=4,# 平衡短语和全局特征
            epochs=15,
            vector_size=32, # 复杂特征需要更高维度
            min_count=1
        )
    outdf = []
    for line in tqdm.tqdm(list(data[i].values)):
        sumarr = 0
        sl = line.split(',')
        for l in sl:
            sumarr = sumarr + model.wv[str(l)]
        outdf.append(sumarr/len(sl))
    tmp = pd.DataFrame(outdf)
    tmp.columns = ['{}'.format(i)+str(j) for j in tmp.columns]
    train_w2vDF = tmp[:train.shape[0]]
    test_w2vDF = tmp[train.shape[0]:].reset_index(drop=True)
    train_w2vDF.to_pickle('./data/train_w2vDF_{}.pkl'.format(i))
    test_w2vDF.to_pickle('./data/test_w2vDF_{}.pkl'.format(i))