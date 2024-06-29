import os
import torch
import models
import numpy as np
import os.path as osp
import torch.optim as optim
from dataloader.dataset import EnoseDataset
from torch.utils.data import DataLoader
from losses.base import TrainingLoss
from utils.common import get_valid_args, ts2np
from utils.utils import cal_metrics, get_confusion_matrix, plot_confusion_matrix, plot_features, plot_loss

class BaseTrainer():
    """
    训练器的基类
    """
    def __init__(self, cfgs):
        """
            依据配置文件构建训练器, 包括: 模型, 数据加载器, 损失函数, 优化器, 学习率调整器, 训练参数等

            下面说明一下BaseTrainer的设计逻辑:
                1, 通过配置文件构建训练器, 配置文件中包含了训练器的所有参数, 包括: 模型, 数据加载器, 损失函数, 优化器, 学习率调整器, 训练参数等
                2, train()方法用于训练模型, 训练过程中会在测试集上进行测试, 并保存最好的模型.
                3, test()方法用于在测试集上测试模型, 并可视化混淆矩阵和特征. 
                4, save()方法用于保存模型
                5, load()方法用于加载模型
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.trainer_cfg = cfgs['trainer_cfg']

        # 创建数据加载器
        self.train_loader, self.test_loader = self.build_loaders(cfgs['data_cfg'], cfgs['dataloader_cfg'])
        
        self.model = self.build_model(cfgs['model_cfg']).to(self.device)
        self.model_name = cfgs['model_cfg']['model_name']  
        
        # 创建损失函数, 优化器, 学习率调整器
        self.criterion = TrainingLoss(cfgs['loss_cfg'])
        self.optimizer = self.get_optimizer(cfgs['optimizer_cfg'])
        self.lr_scheduler = self.get_scheduler(cfgs['scheduler_cfg'])

        # 训练有关参数
        self.start_epoch = 0
        # 最佳测试指标, 用于保存最好的模型
        self.best_test_metric = self.trainer_cfg['test_metric_threshold']
       
        self.epochs = self.trainer_cfg['epochs']
        self.test_metric_threshold = self.trainer_cfg['test_metric_threshold']
        # 早停参数
        self.early_stop = self.trainer_cfg['early_stop'] 

        self.trainer_name = self.trainer_cfg['type']
        self.target_class_idx = self.trainer_cfg['target_class_idx']

        self.save_dir = osp.join(self.trainer_cfg['save_dir'], self.model_name, self.trainer_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        restore_hint = self.trainer_cfg['restore_hint']
        self.load(restore_hint)

    def build_loaders(self, data_cfg, loader_cfg):
        valid_dataset_args = get_valid_args(EnoseDataset, data_cfg)
        train_dataset = EnoseDataset(flag='train', **valid_dataset_args)
        test_dataset = EnoseDataset(flag='test', **valid_dataset_args)

        
        valid_loader_args = get_valid_args(DataLoader, loader_cfg)
        train_loader = DataLoader(train_dataset, shuffle=True, **valid_loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, **valid_loader_args)

        return train_loader, test_loader

    def build_model(self, model_cfg):
        Model = getattr(models, model_cfg['type'])
        valid_model_args = get_valid_args(Model, model_cfg, ['type', 'model_name'])
        model = Model(**valid_model_args)
        return model
    
    def get_optimizer(self, optimizer_cfg):
        # 注意这里在统计参数的时候, 不要忽略在损失函数中的参数, 例如: center loss中的center参数
        Optimizer = getattr(optim, optimizer_cfg['type'])
        valid_arg = get_valid_args(Optimizer, optimizer_cfg, ['type'])
        model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        loss_params = list(filter(lambda p: p.requires_grad, self.criterion.parameters()))
        params = model_params + loss_params
        optimizer = Optimizer(params, **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        Scheduler = getattr(optim.lr_scheduler, scheduler_cfg['type'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['type'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def train(self):
        self.model.train()
        train_loss_list = []
        test_loss_list = []
        no_improvement = 0
        for i in range(self.start_epoch+1, self.epochs+1):
            print(f'############ Epoch: {i} start ###############')
            batch_train_loss = []
            preds = []
            gts = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                logits, feats = self.model(data)
                loss, loss_info = self.criterion(logits, feats, target)
                main_logits = logits[0] if isinstance(logits, list) else logits # 主分类器的输出, 应对TCFusion模型
                pred = main_logits.argmax(dim=1, keepdim=True)
                preds.append(ts2np(pred))
                gts.append(ts2np(target))
                loss.backward()
                self.optimizer.step()
                batch_train_loss.append(loss.item())

            self.lr_scheduler.step() # 调整学习率

            train_loss = np.average(batch_train_loss)
            preds, gts = np.concatenate(preds), np.concatenate(gts)
            train_metrics = cal_metrics(gts, preds, target_cls_id=self.target_class_idx, stage='Training')
            test_loss, test_metrics = self.test(model_epoch=i, is_testing=False) # 在训练阶段进行测试
            test_average = test_metrics['average']
            
            if test_average > self.best_test_metric:
                self.best_test_metric = test_average
                self.save(i, test_average, test_best=True) # 保存最好的模型
                no_improvement = 0
            else:
                no_improvement += 1

            print(f'Epoch: {i}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            for key, value in train_metrics.items(): # 打印训练集和测试集的指标
                print(f'Training {key}: {value:.4f}, Test {key}: {test_metrics[key]:.4f}')

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            
            print() # 换行
            
            if no_improvement >= self.early_stop:
                print(f'Early stopping at epoch {i}')
                break
        
        # 保存训练过程中的loss曲线
        plot_loss(train_loss_list, test_loss_list, self.save_dir)

        # 加载表现最好的模型, 并测试
        self.load(epoch=0)
        self.test(is_testing=True)

    def test(self, model_epoch=None, is_testing=True):
        self.model.eval()
        total_loss = []
        preds = []
        gts = []
        feats_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits, feats = self.model(data)
                loss, loss_info = self.criterion(logits, feats, target)
                main_logits = logits[0] if isinstance(logits, list) else logits # 主分类器的输出, 应对TCFusion模型
                pred = main_logits.argmax(dim=1, keepdim=True)
                feats_list.append(ts2np(feats))
                preds.append(ts2np(pred))
                gts.append(ts2np(target))
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        preds, gts = np.concatenate(preds), np.concatenate(gts)
        metrics = cal_metrics(gts, preds, target_cls_id=self.target_class_idx, stage='Test')

        vis_threshold = self.test_metric_threshold

        # 对于最佳模型或者在测试阶段, 可视化混淆矩阵和特征
        if metrics['average'] > vis_threshold or is_testing:
            model_epoch = self.start_epoch if model_epoch is None else model_epoch
            for key, value in metrics.items():
                print(f'Test {key} on epoch {model_epoch}: {value:.4f}')
            confusion_matrix = get_confusion_matrix(gts, preds)
            plot_confusion_matrix(confusion_matrix, self.save_dir, save_name=f'epoch_{model_epoch}_test_confusion_matrix.jpg')
            # 可视化特征
            feats_list = np.concatenate(feats_list, axis=0)
            plot_features(feats_list, gts, self.save_dir, save_name=f'epoch_{model_epoch}_test_features.jpg')
        
        self.model.train()
        return total_loss, metrics
    
    
    def save(self, epoch, test_average, test_best=False):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'test_average': test_average,
        }
        print(f'Saving model at epoch {epoch} with test average metric: {test_average:.4f} on test set')
        if test_best:  
            torch.save(ckpt, f'{self.save_dir}/best_test_model.pt')
        torch.save(ckpt, f'{self.save_dir}/model_epoch_{epoch}.pt')
        

    def load(self, epoch=0):
        if epoch == 0:
            path = os.path.join(self.save_dir, 'best_test_model.pt')
        else:
            path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch = ckpt['epoch']
            self.best_test_metric = ckpt['test_average']
            print(f"Resuming checkponit from epoch: {self.start_epoch}")
        else:
            print(f'No checkpoint found at {path}, starting from scratch')
    



