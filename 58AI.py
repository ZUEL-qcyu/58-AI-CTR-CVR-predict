import math
import numpy as np
import torch.optim
import tqdm
import gc
from DCN import DCN
from MMoE import MMoE
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

train = pd.read_feather('./data/train.feather')
# print(train['click'].value_counts())
# print(train['send'].value_counts())
test = pd.read_feather('./data/test.feather')
# print(train.columns.tolist())
# print(train.shape)
# print(test.columns.tolist())
# print(test.shape)
click = train['click']
send = train['send']
#职位亮点
train['resume_light_counts']  = train['resume_light_1'] + train['resume_light_2'] + train['resume_light_3']
test['resume_light_counts']  = test['resume_light_1'] + test['resume_light_2'] + test['resume_light_3']
#职位点击总数
train['click_job_count'] = train['click_job_id'].apply(lambda  x:  len(x.split(',')))
test['click_job_count'] = test['click_job_id'].apply(lambda  x:  len(x.split(',')))
#职位薪资与期望薪资差异
train['salary_diff'] = abs(train['salary'] - train['hope_salary'])
test['salary_diff'] = abs(test['salary'] - test['hope_salary'])
#访问时刻*职位更新时刻
train['job_cross_time'] = np.sin(train['hour'])*np.sin(train['job_update_time'])
test['job_cross_time'] = np.sin(test['hour'])*np.sin(test['job_update_time'])
#用户年龄 * 职业工作年限
train['uj_age'] = train['work_age']*train['user_age']
test['uj_age'] = test['work_age']*test['user_age']

#学历交叉
train['edu_cross'] = train['edu_requirment']*train['edu_level']
test['edu_cross'] = test['edu_requirment']*test['edu_level']

#福利累加
welfare = ['welfare1','welfare2','welfare3','welfare4','welfare5','welfare6','welfare7','welfare8','welfare9','welfare10','welfare11','welfare12','welfare13']
for i in range(len(welfare)):
    if i==0:
        train['welfare_add']  = train[welfare[i]]
        test['welfare_add']  = test[welfare[i]]
    else:
        train['welfare_add']  += train[welfare[i]]
        test['welfare_add']  += test[welfare[i]]

train_w2vDF = pd.read_pickle('./data/train_w2vDF_user_profile.pkl')
test_w2vDF = pd.read_pickle('./data/test_w2vDF_user_profile.pkl')
train_w2vDF_click_job = pd.read_pickle('./data/train_w2vDF_click_job_id.pkl')
test_w2vDF_click_job = pd.read_pickle('./data/test_w2vDF_click_job_id.pkl')
train_w2vDF_jobs_feature = pd.read_pickle('./data/train_w2vDF_jobs_feature.pkl')
test_w2vDF_jobs_feature = pd.read_pickle('./data/test_w2vDF_jobs_feature.pkl')
train_w2vDF_job_content_label = pd.read_pickle('./data/train_w2vDF_job_content_label.pkl')
test_w2vDF_job_content_label = pd.read_pickle('./data/test_w2vDF_job_content_label.pkl')
#添加此特征后需在模型参数修改稠密特征维度，此处每个embedding为8维
train = pd.concat([train,train_w2vDF,train_w2vDF_click_job,train_w2vDF_jobs_feature,train_w2vDF_job_content_label],axis=1)
test = pd.concat([test,test_w2vDF,test_w2vDF_click_job,test_w2vDF_jobs_feature,test_w2vDF_job_content_label],axis=1)

drop_columns = ['user_profile', 'click_job_id','jobs_feature','location','page','job_title_label', 'job_content_label']
drop_columns1 = ['user_profile', 'click_job_id','jobs_feature','job_title_label', 'job_content_label']

train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns1, axis=1, inplace=True)

cols_to_convert = ['job_click_rate','job_send_rate', 'uv_click_rate', 'uv_send_rate']
train[cols_to_convert] = train[cols_to_convert].astype(float)
test[cols_to_convert] = test[cols_to_convert].astype(float)

#标准化

col = [tmp_col for tmp_col in train.columns if tmp_col not in ['click', 'send','hour']]

scaler = StandardScaler()

# 使用训练集拟合scaler并转换训练集
train[col] = scaler.fit_transform(train[col])
test[col] = scaler.transform(test[col])


X_train = train[train['hour']<22][col].reset_index(drop=True)
X_val = train[train['hour']>=22][col].reset_index(drop=True)
y_train = train[train['hour']<22][['click', 'send']].reset_index(drop=True)
y_val = train[train['hour']>=22][['click', 'send']].reset_index(drop=True)

X_train = reduce_mem(X_train)
X_val = reduce_mem(X_val)
test = reduce_mem(test)
object_columns = X_train.select_dtypes(include=['object']).columns
#print("Object 类型的列: ", object_columns)


X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values)
X_val_tensor = torch.FloatTensor(X_val.values)
y_val_tensor = torch.FloatTensor(y_val.values)



# 创建Dataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)


batch_size = 2048
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 训练集必须打乱
    num_workers=0,  # 多进程加载
    pin_memory=True  # 加速GPU传输
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size * 2,  # 验证集可用更大batch
    shuffle=False,
    num_workers=0
)


input_dim = X_train.shape[1]
model = MMoE(input_dim=input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cuda:0'
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr= 0.0003)

ctr_criterion = nn.BCELoss()
cvr_criterion = nn.BCELoss()
alpha = 0.5

best_auc = 0
patience = 0
epochs = 50
for epoch in range(epochs):
    model.train()

    epoch_ctr_loss, epoch_cvr_loss = 0, 0
    #train_loader_tqdm = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{200}", leave=True)
    for batch_idx, (batch_input, batch_labels) in enumerate(train_loader):
        batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
        output = model(batch_input)
        CTR, CVR = output[0], output[1]

        click_labels = batch_labels[:, 0]  # click标签列
        send_labels = batch_labels[:, 1]  # send标签列
        #print(f"Label range: {batch_labels.min().item()} to {batch_labels.max().item()}")
        ctr_loss = ctr_criterion(CTR.view(-1), click_labels)

        cvr_mask = (click_labels == 1)
        valid_cvr_pred = CVR.view(-1)[cvr_mask]
        valid_send_labels = send_labels[cvr_mask]

        if len(valid_send_labels) > 0:  # 避免无有效样本时计算报错
            cvr_loss = cvr_criterion(valid_cvr_pred, valid_send_labels)
        else:
            cvr_loss = torch.tensor(0.0, device=device)

        #cvr_loss = cvr_criterion(CVR.view(-1), batch_labels[:, 1])
        # print(ctr_loss)
        # print(cvr_loss)
        total_loss = ctr_loss + alpha*cvr_loss  # 可以加权：α*ctr_loss + β*cvr_loss

        # 反向传播和优化
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        epoch_ctr_loss += ctr_loss.item()
        epoch_cvr_loss += cvr_loss.item() if len(valid_send_labels) > 0 else 0

        # # 更新进度条信息
        # train_loader_tqdm.set_postfix({
        #     "CTR Loss": f"{ctr_loss.item():.4f}",
        #     "CVR Loss": f"{cvr_loss.item():.4f}" if len(valid_send_labels) > 0 else "N/A"
        # })


    model.eval()
    val_ctr_loss, val_cvr_loss = 0, 0
    ctr_preds, ctr_targets = [], []
    cvr_preds, cvr_targets = [], []
    epoch_val_loss = 0
    #val_loader_tqdm = tqdm.tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False)
    with torch.no_grad():
        for batch_idx, (batch_input, batch_labels) in enumerate(val_loader):
            batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
            click_labels = batch_labels[:, 0]
            send_labels = batch_labels[:, 1]

            output = model(batch_input)
            CTR, CVR = output[0], output[1]

            ctr_loss = ctr_criterion(CTR.view(-1), click_labels)
            val_ctr_loss += ctr_loss.item()

            cvr_mask = (click_labels == 1)
            valid_cvr_pred = CVR.view(-1)[cvr_mask]
            valid_send_labels = send_labels[cvr_mask]

            if len(valid_send_labels) > 0:
                cvr_loss = cvr_criterion(valid_cvr_pred, valid_send_labels)
                val_cvr_loss += cvr_loss.item()

                # 收集CVR预测结果用于计算AUC等指标
                cvr_preds.extend(valid_cvr_pred.cpu().numpy())
                cvr_targets.extend(valid_send_labels.cpu().numpy())

            # 收集CTR预测结果
            ctr_preds.extend(CTR.cpu().numpy())
            ctr_targets.extend(click_labels.cpu().numpy())

            # --- 计算验证指标 ---
        avg_val_ctr_loss = val_ctr_loss / len(val_loader)
        avg_val_cvr_loss = val_cvr_loss / len(val_loader) if len(cvr_targets) > 0 else 0

        # AUC
        ctr_auc = roc_auc_score(ctr_targets, ctr_preds)
        cvr_auc = roc_auc_score(cvr_targets, cvr_preds) if len(cvr_targets) > 0 else 0
        if ctr_auc > best_auc:
            best_auc = ctr_auc
            torch.save(model.state_dict(), 'model_weights.pth')
            patience = 0


        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train CTR Loss: {epoch_ctr_loss / len(train_loader):.4f} | "
              f"Train CVR Loss: {epoch_cvr_loss / len(train_loader):.4f} | "
              f"Val CTR Loss: {avg_val_ctr_loss:.4f} (AUC: {ctr_auc:.4f}) | "
              f"Val CVR Loss: {avg_val_cvr_loss:.4f} (AUC: {cvr_auc:.4f})")

        patience += 1

        if patience>5:break


model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
model.eval()

test = torch.from_numpy(test.values).float()
test_res = model(test)
df = pd.DataFrame(test_res)
df.to_csv('test_res.csv', index=False)

