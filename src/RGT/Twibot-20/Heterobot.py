from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from layer import RGTLayer
import pytorch_lightning as pl
from torch import nn
import torch
from Dataset import BotDataset
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir


class RGTDetector(pl.LightningModule):
    def __init__(self, args):
        super(RGTDetector, self).__init__()
        self.edge_index = torch.load(args.path + "edge_index.pt", map_location="cuda")
        self.edge_type = torch.load(args.path + "edge_type.pt", map_location="cuda")
        self.label = torch.load(args.path + "label.pt", map_location="cuda")
  
        self.lr = args.lr
        self.l2_reg = args.l2_reg

        self.cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cuda")
        self.prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cuda")
        self.tweet_features = torch.load(args.path + "tweets_tensor.pt", map_location="cuda")
        self.des_features = torch.load(args.path + "des_tensor.pt", map_location="cuda")

        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGT_layer1 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()
        # 新增：初始化空列表，用于存储 test_step 中的嵌入和标签
        self.test_embeddings_list = []  # 存每次批次的测试集节点嵌入
        self.test_labels_list = []      # 存每次批次的测试集节点真实标签
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        train_batch = train_batch.squeeze(0)

        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
        
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        user_features = self.ReLU(self.RGT_layer1(user_features, self.edge_index, self.edge_type))
        user_features = self.ReLU(self.RGT_layer2(user_features, self.edge_index, self.edge_type))

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features[train_batch])
        loss = self.CELoss(pred, self.label[train_batch])

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            val_batch = val_batch.squeeze(0)

            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, self.edge_index, self.edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, self.edge_index, self.edge_type))

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features[val_batch])
            # print(pred.size())
            pred_binary = torch.argmax(pred, dim=1)
            
            # print(self.label[val_batch].size())

            acc = accuracy_score(self.label[val_batch].cpu(), pred_binary.cpu())
            f1 = f1_score(self.label[val_batch].cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc)
            self.log("val_f1", f1)

            print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            test_batch = test_batch.squeeze(0)
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, self.edge_index, self.edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, self.edge_index, self.edge_type))

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features[test_batch])
            
            pred_binary = torch.argmax(pred, dim=1)

            acc = accuracy_score(self.label[test_batch].cpu(), pred_binary.cpu())
            f1 = f1_score(self.label[test_batch].cpu(), pred_binary.cpu())
            precision =precision_score(self.label[test_batch].cpu(), pred_binary.cpu())
            recall = recall_score(self.label[test_batch].cpu(), pred_binary.cpu())
            auc = roc_auc_score(self.label[test_batch].cpu(), pred[:,1].cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", auc)

            print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))
            # 新增：保存当前批次的测试集嵌入和标签到列表
            # user_features[test_batch]：当前批次测试集节点的嵌入
            # self.label[test_batch]：当前批次测试集节点的真实标签
            self.test_embeddings_list.append(user_features[test_batch].cpu())  # 移到CPU，避免GPU内存占用
            self.test_labels_list.append(self.label[test_batch].cpu())

    def get_test_embeddings_labels(self):
        """将分批保存的测试集嵌入和标签拼接成完整 tensor，供可视化调用"""
        # 拼接所有批次的嵌入（从列表→tensor）
        all_test_embeddings = torch.cat(self.test_embeddings_list, dim=0)
        # 拼接所有批次的标签（从列表→tensor）
        all_test_labels = torch.cat(self.test_labels_list, dim=0)
        # 清空列表（避免下次测试重复存储）
        self.test_embeddings_list = []
        self.test_labels_list = []
        return all_test_embeddings, all_test_labels

    def visualize_clear(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # 1. 获取保存的测试集嵌入和标签
        embeddings, labels = self.get_test_embeddings_labels()
        if torch.is_tensor(embeddings):
            embeddings = embeddings.numpy()
        if torch.is_tensor(labels):
            labels = labels.numpy()

        # 2. T-SNE 降维
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
        plt.savefig('tsne_test_set_RGT_end5.pdf', format='pdf', bbox_inches='tight')
        plt.close()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


parser = argparse.ArgumentParser(description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="/home/gaojie/BotDGT-master/data/Twibot-20/processed_data/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--trans_head", type=int, default=2, help="description channel")
parser.add_argument("--semantic_head", type=int, default=2, help="description channel")
parser.add_argument("--batch_size", type=int, default=128, help="description channel")
parser.add_argument("--epochs", type=int, default=50, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    if args.random_seed != None:
        pl.seed_everything(args.random_seed)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    train_dataset = BotDataset(name="train")
    valid_dataset = BotDataset(name="valid")
    test_dataset = BotDataset(name="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = RGTDetector(args)
    # trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1, callbacks=[checkpoint_callback])
    trainer = pl.Trainer(
        accelerator="gpu",  # 指定使用GPU加速
        devices=1,  # 使用1块GPU（替代原来的gpus=1）
        num_nodes=1,
        max_epochs=args.epochs,
        precision=16,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, train_loader, valid_loader)

    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])

    best_model = RGTDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)
    # 测试完成后，调用可视化函数（自动获取保存的嵌入和标签）
    # best_model.visualize_clear()