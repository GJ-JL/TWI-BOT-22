import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import null_metrics, calc_metrics, is_better
import torch
from dataset import get_train_data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import torch.nn as nn
from model import BotGAT, BotGCN, BotRGCN
from sklearn.manifold import TSNE

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-20')
parser.add_argument('--mode', type=str, default='GAT')
parser.add_argument('--visible', type=bool, default=True)  # 改为True，便于观察中间输出
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_up', type=int, default=40)
parser.add_argument('--lr', type=float, default=3e-4)  # 适当提高学习率
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.3)
args = parser.parse_args()

dataset_name = args.dataset
mode = args.mode
visible = args.visible

assert mode in ['GCN', 'GAT', 'RGCN']
assert dataset_name in ['cresci-2015', 'Twibot-20', 'Twibot-22']

# -------------------------- 关键：验证数据集标签和索引正确性 --------------------------
data = get_train_data(dataset_name)
print(f"数据集基本信息：总样本数={data.num_nodes}，标签值范围={torch.unique(data.y)}")
print(f"训练集样本数={len(data.train_idx)}，验证集={len(data.val_idx)}，测试集={len(data.test_idx)}")
print(f"训练集标签分布：0类={torch.sum(data.y[data.train_idx]==0)}, 1类={torch.sum(data.y[data.train_idx]==1)}")
print(f"测试集标签分布：0类={torch.sum(data.y[data.test_idx]==0)}, 1类={torch.sum(data.y[data.test_idx]==1)}")
# 若测试集1类数量为0，直接报错（根本原因）
assert torch.sum(data.y[data.test_idx]==1) > 0, "测试集无少数类样本，请检查数据集划分！"

hidden_dim = args.hidden_dim
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay
max_epoch = args.max_epoch
batch_size = args.batch_size
no_up = args.no_up

# 全局变量：保存测试集高维嵌入和标签
test_high_dim_embeddings = []
test_labels = []


def visualize_test_embeddings():
    if len(test_high_dim_embeddings) == 0 or len(test_labels) == 0:
        print("Error: 未获取到测试集高维嵌入和标签！")
        return
    embeddings = np.concatenate(test_high_dim_embeddings, axis=0)
    labels = np.concatenate(test_labels, axis=0)
    print(f"可视化输入：嵌入形状={embeddings.shape}，标签分布=0类{np.sum(labels==0)},1类{np.sum(labels==1)}")

    # 2. T-SNE 降维（更紧凑参数）
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

    plt.savefig('tsne_test_set_GAT_end5.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"可视化完成：测试集节点数={len(labels)}")


def forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader):
    model.train()
    all_label = []
    all_pred = []  # 存储logits，供calc_metrics使用
    ave_loss = 0.0
    cnt = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_batch = batch.batch_size
        # 接收模型返回的分类输出和高维嵌入（训练阶段忽略嵌入）
        out, _ = model(
            batch.des_embedding,
            batch.tweet_embedding,
            batch.num_property_embedding,
            batch.cat_property_embedding,
            batch.edge_index,
            batch.edge_type
        )
        label = batch.y[:n_batch]
        out = out[:n_batch]
        # 收集标签和预测（logits）
        all_label.append(label.detach().cpu())
        all_pred.append(out.detach().cpu())
        # 计算损失
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch  # 加权累积损失
        cnt += n_batch
        loss.backward()
        optimizer.step()
    # 修复：损失只除以一次总样本数
    ave_loss /= cnt
    # 拼接所有批次的标签和预测
    all_label = torch.cat(all_label, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    # 计算指标
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = f'Epoch-{epoch} train loss: {ave_loss:.6f}' + plog
    if visible:
        print(plog)
    # 验证
    val_metrics = validation(epoch, 'validation', model, loss_fn, val_loader)
    return val_metrics


@torch.no_grad()
def validation(epoch, name, model, loss_fn, loader):
    model.eval()
    all_label = []
    all_pred = []  # 存储logits
    ave_loss = 0.0
    cnt = 0.0
    is_test_phase = (name == 'test')
    if is_test_phase:
        global test_high_dim_embeddings, test_labels
        test_high_dim_embeddings.clear()
        test_labels.clear()

    for batch in loader:
        batch = batch.to(device)
        n_batch = batch.batch_size
        out, high_dim_emb = model(
            batch.des_embedding,
            batch.tweet_embedding,
            batch.num_property_embedding,
            batch.cat_property_embedding,
            batch.edge_index,
            batch.edge_type
        )
        # 测试阶段保存嵌入和标签
        if is_test_phase:
            test_high_dim_embeddings.append(high_dim_emb[:n_batch].cpu().numpy())
            test_labels.append(batch.y[:n_batch].cpu().numpy())
        # 收集标签和预测
        label = batch.y[:n_batch]
        out = out[:n_batch]
        all_label.append(label.cpu())
        all_pred.append(out.cpu())
        # 计算损失
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
    # 计算平均损失
    ave_loss /= cnt
    # 拼接所有批次
    all_label = torch.cat(all_label, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    # 打印当前阶段的标签和预测分布（关键调试信息）
    pred_labels = torch.argmax(all_pred, dim=1)
    print(f"{name}集标签分布：0类={torch.sum(all_label==0)}, 1类={torch.sum(all_label==1)}")
    print(f"{name}集预测分布：0类={torch.sum(pred_labels==0)}, 1类={torch.sum(pred_labels==1)}")
    # 计算指标
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = f'Epoch-{epoch} {name} loss: {ave_loss:.6f}' + plog
    if visible:
        print(plog)
    return metrics


def train():
    data.edge_index = data.edge_index.contiguous()
    save_dir = '/home/gaojie/TwiBot-22-master/src/GCN_GAT'
    os.makedirs(save_dir, exist_ok=True)

    # 数据加载器
    train_loader = NeighborLoader(
        data,
        num_neighbors=[256] * 2,
        batch_size=batch_size,
        input_nodes=data.train_idx,
        shuffle=True
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=[256] * 2,
        batch_size=batch_size,
        input_nodes=data.val_idx
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=[256] * 2,
        batch_size=batch_size,
        input_nodes=data.test_idx
    )

    # 初始化模型
    if mode == 'GAT':
        model = BotGAT(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_prop_size=data.num_property_embedding.shape[-1],
            cat_prop_size=data.cat_property_embedding.shape[-1]
        ).to(device)
    elif mode == 'GCN':
        model = BotGCN(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_prop_size=data.num_property_embedding.shape[-1],
            cat_prop_size=data.cat_property_embedding.shape[-1]
        ).to(device)
    elif mode == 'RGCN':
        model = BotRGCN(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_prop_size=data.num_property_embedding.shape[-1],
            cat_prop_size=data.cat_property_embedding.shape[-1],
            num_relations=data.edge_type.max().item() + 1
        ).to(device)
    else:
        raise KeyError

    # 关键：使用带权重的损失函数解决类别不平衡
    train_labels = data.y[data.train_idx]
    class_0_count = torch.sum(train_labels == 0)
    class_1_count = torch.sum(train_labels == 1)
    class_weights = torch.tensor(
        [class_1_count / (class_0_count + class_1_count),  # 0类权重（反比于样本数）
         class_0_count / (class_0_count + class_1_count)], # 1类权重（反比于样本数）
        device=device, dtype=torch.float32
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # 带权重的损失

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_metrics = null_metrics()
    best_state_dict = None
    pbar = tqdm(range(max_epoch), ncols=0)
    cnt = 0

    for epoch in pbar:
        val_metrics = forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader)
        if is_better(val_metrics, best_val_metrics):
            best_val_metrics = val_metrics
            best_state_dict = model.state_dict()
            cnt = 0
        else:
            cnt += 1
        pbar.set_postfix_str(f'val acc {val_metrics["acc"]:.4f} no up cnt {cnt}')
        if cnt == no_up:
            break

    # 测试
    model.load_state_dict(best_state_dict)
    test_metrics = validation(max_epoch, 'test', model, loss_fn, test_loader)

    # 保存模型
    model_filename = f"{dataset_name}_{mode}_{test_metrics['acc']:.4f}.pt"
    save_path = os.path.join(save_dir, model_filename)
    torch.save(best_state_dict, save_path)
    print(f"模型已保存至: {save_path}")

    # 打印测试指标
    print("\n测试集最终指标：")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    # 可视化
    visualize_test_embeddings()


if __name__ == '__main__':
    train()