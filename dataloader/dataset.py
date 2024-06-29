import os
import sys
sys.path.append('./')
import torch
import random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.common import is_list, is_ndarray
from utils.utils import plot_features
import dataloader.tsgm_augmentations as tsgm_augs


def scale_data(data, scaler='Minmax'):
    scaler = StandardScaler() if scaler == 'Standard' else MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

class EnoseDataset(Dataset):
    """
    数据描述:
        数据集中主要有四类数据:
        VOC: 1-11, 每个类别都是一个生物标记物. 样本大小依次为:[188, 153, 160, 148, 189, 205, 209, 221, 187, 205, 180]
        Water: 样本大小:27
        UBC: 样本大小:76
        Healthy: 样本大小:18
    
    数据预处理:
        每个.mat文件代表一个具有可变数量样本的类. 每个样本都是一个具有可变长度的序列数据. 
        为了简单起见, 我们将每个样本的数据剪切到固定长度（例如, 65). 
        对于VOCs, 数据将被随机分割为训练/测试集, 比例为4:1. 
        对于Water/UBC/Healthy样本, 将随机选择5个样本作为训练样本, 其余的将用作测试样本. 
        这5个训练样本将通过生成模型增强到100个样本. 
    
    数据缩放（可选）:
        - 对于每个样本, 我们将其数据进行缩放, 使其均值为0, 方差为1. StandardScaler将用于缩放数据. 

    Notes:
        1. 所有的.mat文件都应该存放在对应的主类文件夹中
        2. .mat文件的命名格式为: ClassName_label.mat, 其中ClassName为类名, label为类别标签
        3. data_cfg包含了需要读取的数据, 包含主类文件

    有关数据集加载的逻辑解释: 
        1, 读取数据, 并构建标签映射关系. 标签映射关系是类别到连续整数的映射, 
                如[0, 1, 3, 4, 5]转成{0:0, 1:1, 3:2, 4:3, 5:4}
        2, 依据标签映射关系, 读取数据并随机划分为训练集和测试集, 返回的标签是映射后的标签. 
        5, 数据有些需要进行数据增强, 例如VOCs, 水, UBC, 健康人, 只能对训练集进行数据增强, 不能对测试集进行数据增强. 
    """
    all_data = None # 所有数据, 包含训练集和测试集
    label_mapping = None # 标签映射关系

    def __init__(self, data_dir, dataset_cfg=None, flag='train', seed=0, scale=True):
        """
        Parameters:
            data_dir (str): 数据集所在的目录
            dataset_cfg (dict): 数据集的配置文件, 包括主类文件
            flag (str): 用于指示读取数据集的类型 (train/test)
            seed (int): 随机种子, 用于划分训练集和测试集
            scale (bool): 是否对数据进行缩放
        """
        self.data_dir = data_dir
        self.dataset_cfg = dataset_cfg
        self.channels = dataset_cfg['channels']
        self.flag = flag
        self.seed = seed
        if EnoseDataset.all_data is None:
            EnoseDataset.all_data = EnoseDataset.load_and_split_data(data_dir, dataset_cfg, seed, scale)

        # 获取数据和标签
        self.data_list, self.label_list = EnoseDataset.get_dataset_by_flag(flag)
        
        if flag == 'train' and self.dataset_cfg['use_tsgm']:
            # 训练集需要进行数据增强
            self.generate_data_by_tsgm()

        assert len(self.data_list) == len(self.label_list), 'The length of data list and label list should be the same!'

    def generate_data_by_tsgm(self):
        """
            生成数据并添加到训练集中, 使用TSGM方法
        """
        # step1: 首先根据配置文件获取TSGM的配置
        cfg_list = self.dataset_cfg['tsgm_cfg']
        save_dir = self.dataset_cfg['tsgm_save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        all_new_data, all_new_label = [], []
        # step2: 依次执行TSGM方法
        for cfg in cfg_list:
            aug_type = cfg['type']
            Aug_model = getattr(tsgm_augs, aug_type)
            aug_model = Aug_model()
            class_id_list = cfg['class_id_list'] # 需要生成数据的类别
            gen_num_list = cfg['gen_num_list'] # 生成的数据数量
            gen_params = cfg['gen_params'] # 生成数据的参数
            gen_params = {} if gen_params is None else gen_params
            assert len(class_id_list) == len(gen_num_list), 'The length of class_id_list and gen_num_list should be the same!'
            for i in range(len(class_id_list)):
                class_id = class_id_list[i]
                gen_num = gen_num_list[i]
                # step2.0: 查看是否存在生成好的数据
                data_path = os.path.join(save_dir, f'{class_id}_{gen_num}_{aug_type}_data.npy')
                label_path = os.path.join(save_dir, f'{class_id}_{gen_num}_{aug_type}_label.npy')
                if os.path.exists(data_path) and os.path.exists(label_path):
                    new_data = np.load(data_path)
                    new_label = np.load(label_path) 
                    if len(new_label.shape) == 2:
                        new_label = new_label.reshape(-1)
                else:
                    # step2.1: 获取需要生成数据的类别的数据
                    source_data, source_label = self.get_data_by_labels([class_id])
                    # step2.2: 生成数据
                    print(f'Generate {gen_num} samples with label {class_id}')
                    new_data, new_label = aug_model.generate(X=source_data, y=source_label, 
                                                                    n_samples=gen_num, **gen_params)
                    if len(new_label.shape) == 2:
                        new_label = new_label.reshape(-1)
                    # step2.3: 保存生成的数据
                    np.save(data_path, new_data)
                    np.save(label_path, new_label)
                all_new_data.append(new_data)
                all_new_label.append(new_label)
        # step2.4: 添加数据到训练集中
        all_new_data = np.concatenate(all_new_data, axis=0)
        all_new_label = np.concatenate(all_new_label, axis=0)
        self.vis_tsgm_data()
        self.add_samples(all_new_data, all_new_label)
        print(f'Add {len(all_new_data)} samples to train dataset!')
        
    def vis_tsgm_data(self):
        """
            可视化TSGM生成的数据
        """
        save_dir = self.dataset_cfg['tsgm_save_dir']
        file_list = os.listdir(save_dir)
        aug_class_list = []
        agu_info_list = []
        for file_name in file_list:
            if 'data.npy' in file_name:
                class_id = int(file_name.split('_')[0])
                aug_num = int(file_name.split('_')[1])
                aug_type = file_name.split('_')[2]
                if class_id not in aug_class_list:
                    aug_class_list.append(class_id)
                agu_info_list.append((class_id, aug_num, aug_type))
        agu_label_mapping = {}
        for class_id in aug_class_list:
            class_ori_data, class_ori_label = self.get_data_by_labels([class_id])
            class_ori_label = class_ori_label * 0 # 原始数据的标签都是0
            class_gen_data, class_gen_label = [], []
            for aug_id, aug_num, aug_type in agu_info_list:
                if class_id == aug_id:
                    data_path = os.path.join(save_dir, f'{class_id}_{aug_num}_{aug_type}_data.npy')  
                    new_data = np.load(data_path)
                    if aug_type not in agu_label_mapping.keys():
                        agu_label_mapping[aug_type] = len(agu_label_mapping) + 1
                    new_label = np.ones(new_data.shape[0]) * agu_label_mapping[aug_type]
                    class_gen_data.append(new_data)
                    class_gen_label.append(new_label)
            class_gen_data = np.concatenate(class_gen_data, axis=0)
            class_gen_label = np.concatenate(class_gen_label, axis=0)
            class_data = np.concatenate([class_ori_data, class_gen_data], axis=0)
            class_label = np.concatenate([class_ori_label, class_gen_label], axis=0)
            # flatten data
            class_data = class_data.reshape(class_data.shape[0], -1)
            class_label = class_label.astype(np.int32)
            plot_features(class_data, class_label, save_dir, save_name=f'{class_id}_tsgm.jpg', save_features=False)
                
                                

    @staticmethod
    def get_data_by_labels(labels):
        """
            根据标签获取数据
            labels: list, 标签列表, 如[0, 1, 3, 4, 5], 还没有经过标签映射
            return: data_np, label_np
            获取的数据和标签都是numpy数组, 标签是原始标签, 而不是映射后的标签, 如[0, 1, 3, 4, 5]
            标签映射是在add_samples方法中完成的
        """
        train_data_list, train_label_list = EnoseDataset.all_data[0]
        # 先把查询的标签映射到连续整数
        label_mapping = EnoseDataset.label_mapping
        mapped_labels = [label_mapping[label] for label in labels]
        data_list, label_list = [], []
        for data, label in zip(train_data_list, train_label_list):
            if label in mapped_labels:
                data_list.append(data)
                # 为了避免重复使用生成的数据, 这里存储的是原始标签, 而不是映射后的标签
                idx = mapped_labels.index(label)
                org_label = labels[idx] # 原始标签
                label_list.append(org_label)
        return np.array(data_list), np.array(label_list)

    @staticmethod
    def get_label_mapping(label_list):
        """
            获取标签映射, 如[0, 1, 3, 4, 5]转成{0:0, 1:1, 3:2, 4:3, 5:4}, 
            应该在读取完数据之后立刻调用
        """
        unique_labels = sorted(set(label_list))
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        return label_mapping

    @staticmethod
    def get_dataset_by_flag(flag):
        if flag == 'train':
            data_list, label_list = EnoseDataset.all_data[0]
        elif flag == 'test':
            data_list, label_list = EnoseDataset.all_data[1]
        else:
            raise ValueError('Invalid flag! Please use one of the following: train/test, but got {}'.format(flag))
        return data_list, label_list

    @staticmethod
    def load_and_split_data(data_dir, data_cfg, seed=0, scale=True):
        """
            加载数据, 并随机划分为训练集和测试集, 返回的标签是映射后的标签
        """
        data_list, label_list = EnoseDataset.load_all_data(data_dir, data_cfg, scale)
        class_train_samples = data_cfg['class_train_samples']  # 每个类别样本的训练数据
        # 把数据和标签转换为numpy数组
        data_array = np.array(data_list)
        label_array = np.array(label_list)

        # 初始化训练集和测试集
        train_data_list, test_data_list = [], []
        train_label_list, test_label_list = [], []

        # 找到所有的类别标签
        unique_labels = np.unique(label_array)

        # 遍历每个类别, 并将其分割为训练集和测试集
        for label in unique_labels:
            
            # 找到该类别的所有样本的索引
            indices = np.where(label_array == label)[0]

            # 设置随机种子
            np.random.seed(seed)

            # 打乱索引顺序
            np.random.shuffle(indices)

            # 计算训练集的大小
            num_samples = len(indices)
            train_size = int(0.8 * num_samples)  # 80%用于训练

            if label in class_train_samples.keys():
                # 如果该类别的训练样本数量已经给定, 则使用给定的数量
                train_size = class_train_samples[label]
            
            train_indices = indices[:train_size]
            # print(f"train indices for class {label}", train_indices)
            test_indices = indices[train_size:]
            
            # 将数据和标签添加到训练集和测试集中
            train_data_list.extend(data_array[train_indices])
            test_data_list.extend(data_array[test_indices])

            mapping_label = EnoseDataset.label_mapping[label] # 获取映射后的标签

            # 添加映射后的标签
            train_label_list.extend([mapping_label] * len(train_indices)) 
            test_label_list.extend([mapping_label] * len(test_indices))

        # 返回训练集和测试集
        return (train_data_list, train_label_list), (test_data_list, test_label_list)

    @staticmethod
    def load_all_data(data_dir, data_cfg, scale=True):
        """
            加载数据, 使用静态方法是为了避免重复加载数据
            注意这里返回的label是不连续的, 如[0, 1, 3, 4, 5], 还没有经过标签映射
        """
        seq_len = data_cfg['seq_len'] # 序列长度
        # 加载数据
        all_data_list = []
        all_label_list = []
        all_class_list = data_cfg['All'] # 主要类别
        for class_name in all_class_list:
            class_dir = os.path.join(data_dir, class_name)
            file_name_list = data_cfg[class_name] # 该类别下的.mat文件名列表
            for file_name in file_name_list:
                file_path = os.path.join(class_dir, file_name+'.mat')
                data = sio.loadmat(file_path)[file_name] # [N_sample, 1]
                label = int(file_name.split('_')[-1])
                for i in range(data.shape[0]):
                    sample_data = data[i][0][:seq_len].astype(np.float32) # [seq_len, num_channels]
                    sample_data = np.nan_to_num(sample_data) # 将nan替换为0
                    if scale:
                        sample_data = scale_data(sample_data)
                    all_data_list.append(sample_data)
                    all_label_list.append(label)
        # 读取完所有的数据之后, 需要构建标签映射关系
        EnoseDataset.label_mapping = EnoseDataset.get_label_mapping(all_label_list)
        return all_data_list, all_label_list

    
    def add_samples(self, datas, labels):
        """
            添加生成的样本, labels是原始标签, 而不是映射后的标签
            所以这里需要先根据标签映射关系转换标签
        """
        if self.flag != 'train':
            raise ValueError('Only train dataset can add samples!')
        label_mapping = EnoseDataset.label_mapping
        if is_list(datas):
            self.data_list.extend(datas)
            # 根据标签映射关系转换标签
            labels = [label_mapping[label] for label in labels]
            self.label_list.extend(labels)
        elif is_ndarray(datas) :
            if len(datas.shape) == 2: # [seq_len, num_channels]
                self.data_list.append(datas)
                # 根据标签映射关系转换标签
                labels = label_mapping[labels]
                self.label_list.append(labels)
            elif len(datas.shape) == 3: # [N_samples, seq_len, num_channels]
                self.data_list.extend(datas)
                labels = labels.reshape(-1) # [N_samples]
                # 根据标签映射关系转换标签
                labels = [label_mapping[label] for label in labels]
                self.label_list.extend(labels)
        else:
            raise ValueError('Invalid type of datas! Should be list or np.ndarray! Got {}'.format(type(datas)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index] # [seq_len, num_channels]
        data = data[:,self.channels]
        label = self.label_list[index]
        return data.astype(np.float32), torch.tensor(label, dtype=torch.long)
       
 
if __name__ == '__main__':
    from utils.common import config_loader, get_valid_args
       
    config_path = './configs/base.yaml'
    
    cfg = config_loader(config_path)
    data_cfg = cfg['data_cfg']
    data_cfg['dataset_cfg']['All'] = ['UBC', 'Healthy', 'Water', 'VOC']
    data_cfg['dataset_cfg']['use_tsgm'] = True
    data_scale = True
    data_cfg['scale']= data_scale
    valid_dataset_args = get_valid_args(EnoseDataset, data_cfg)
    train_dataset = EnoseDataset(flag='train', **valid_dataset_args)
    selected_classes = [0, 1, 4, 7, 10, 11, 12, 13]
    print('training set')
    train_labels = train_dataset.label_list
    label_unique = np.unique(train_labels, return_counts=True)
    # show count of selected classes
    for index, class_i in enumerate(selected_classes):
        print(f'Class {class_i} count: {label_unique[1][index]}')
    test_dataset = EnoseDataset(flag='test', **valid_dataset_args)
    print('testing set')
    test_labels = test_dataset.label_list
    label_unique = np.unique(test_labels, return_counts=True)
    for index, class_i in enumerate(selected_classes):
        print(f'Class {class_i} count: {label_unique[1][index]}')
        