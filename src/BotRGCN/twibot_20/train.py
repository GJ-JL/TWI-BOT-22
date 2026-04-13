from model import BotRGCN

from Dataset import Twibot22
import torch
from torch import nn
from utils import accuracy, init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd

device = 'cuda:0'
embedding_size, dropout, lr, weight_decay = 32, 0.1, 1e-2, 5e-2


# root = './processed_data/'
root = '/home/gaojie/TwiBot-22-master/src/BotRGCN/twibot_20/processed_data'

dataset=Twibot22(root=root,device=device,process=False,save=False)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

def load_behavior_features(device):
    path = "/home/gaojie/TwiBot-22-master/src/BotRGCN/twibot_20/processed_data/behavior_properties_tensor.pt"
    beh = torch.load(path, map_location="cpu")
    beh = beh.to(device)
    return beh
behavior_prop = load_behavior_features(device)


model=BotRGCN(cat_prop_size=3,embedding_dimension=embedding_size).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)
# 全局变量：存储 test() 输出的高维嵌入和标签（供可视化使用）
global_test_embeddings = None  # 高维嵌入（未降维前）
global_test_labels = None      # 对应测试集标签


def train(epoch):
    model.train()#训练模式
    output, _ = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()#清除梯度
    loss_train.backward()
    optimizer.step()#更新模型参数
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

# def test():#评估模式
#     model.eval()
#     global global_test_embeddings, global_test_labels
#     output, high_dim_embedding = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
#     # 提取测试集的高维嵌入和标签（转CPU+numpy，供可视化使用）
#     global_test_embeddings = high_dim_embedding[test_idx].cpu().numpy()
#     global_test_labels = labels[test_idx].cpu().numpy()
#     loss_test = loss(output[test_idx], labels[test_idx])
#     acc_test = accuracy(output[test_idx], labels[test_idx])
#     output=output.max(1)[1].to('cpu').detach().numpy()
#     label=labels.to('cpu').detach().numpy()
#     f1=f1_score(label[test_idx],output[test_idx])
#     #mcc=matthews_corrcoef(label[test_idx], output[test_idx])
#     precision=precision_score(label[test_idx],output[test_idx])
#     recall=recall_score(label[test_idx],output[test_idx])
#     fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
#     Auc=auc(fpr, tpr)
#     print("Test set results:",
#             "test_loss= {:.4f}".format(loss_test.item()),
#             "test_accuracy= {:.4f}".format(acc_test.item()),
#             "precision= {:.4f}".format(precision),
#             "recall= {:.4f}".format(recall),
#             "f1_score= {:.4f}".format(f1),
#             #"mcc= {:.4f}".format(mcc.item()),
#             "auc= {:.4f}".format(Auc),
#             )
def test():
    # 全局变量：修改外部的高维嵌入和标签
    global global_test_embeddings, global_test_labels
    model.eval()
    with torch.no_grad():  # 关闭梯度计算，节省内存
        # 从模型获取：分类输出（output） + 高维嵌入（high_dim_embedding）
        output, high_dim_embedding = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)

        # 核心修复：添加 .detach() 解除梯度绑定，再转CPU和numpy
        global_test_embeddings = high_dim_embedding[test_idx].detach().cpu().numpy()
        global_test_labels = labels[test_idx].detach().cpu().numpy()  # 标签也建议加 .detach()，避免潜在问题

        # 计算测试指标（原有逻辑不变，仅修复 output_pred 和 label_np 的梯度问题）
        loss_test = loss(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])
        output_pred = output.max(1)[1].detach().cpu().numpy()  # 这里也需要 .detach()
        label_np = labels.detach().cpu().numpy()  # 标签转numpy前加 .detach()

        f1 = f1_score(label_np[test_idx], output_pred[test_idx])
        precision = precision_score(label_np[test_idx], output_pred[test_idx])
        recall = recall_score(label_np[test_idx], output_pred[test_idx])
        fpr, tpr, thresholds = roc_curve(label_np[test_idx], output_pred[test_idx], pos_label=1)
        Auc = auc(fpr, tpr)

        print("\nTest set results:",
              "test_loss= {:.4f}".format(loss_test.item()),
              "test_accuracy= {:.4f}".format(acc_test.item()),
              "precision= {:.4f}".format(precision),
              "recall= {:.4f}".format(recall),
              "f1_score= {:.4f}".format(f1),
              "auc= {:.4f}".format(Auc))
def visualize_clear():
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # 从全局变量获取高维嵌入和标签
        embeddings = global_test_embeddings
        labels = global_test_labels
        if torch.is_tensor(embeddings):
            embeddings = embeddings.numpy()
        if torch.is_tensor(labels):
            labels = labels.numpy()

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=20,  # 类簇更紧凑
            n_iter=6000,
            early_exaggeration=8.0,  # 提高类间分离度
            min_grad_norm=1e-8,
            learning_rate=100
        )
        embeddings_2d = tsne.fit_transform(embeddings)

        # 3. 绘图
        plt.figure(figsize=(9, 7))
        plt.axis('off')

        # 🔵🔴 学术安全色
        blue_color = '#0072B2'
        red_color = '#E41E26'

        # Human 类别
        human_mask = labels == 0
        plt.scatter(
            embeddings_2d[human_mask, 0], embeddings_2d[human_mask, 1],
            c=blue_color,
            alpha=1,
            s=300,  # 大点
            linewidths=1.0,
            edgecolors='white'
        )

        # Bot 类别
        bot_mask = labels == 1
        plt.scatter(
            embeddings_2d[bot_mask, 0], embeddings_2d[bot_mask, 1],
            c=red_color,
            alpha=1,
            s=300,
            linewidths=1.0,
            edgecolors='white'
        )

        # 保存PDF
        plt.savefig('tsne_test_set_BOTRGCN_end5.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    
model.apply(init_weights)

epochs = 50
for epoch in range(epochs):
    train(epoch)
    
test()
# # 4. 可视化高维嵌入（生成PDF）
# print("\n开始生成可视化...")
# visualize_clear()