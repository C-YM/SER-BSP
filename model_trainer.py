import itertools
import time

import torch
import torch.nn as nn
from sklearn.metrics import recall_score

from data_loader import label_MAPPING
from models.BSP import BSP
from models.CNN_LSTM_ATT_SINGLE import CNN_LSTM_ATT_SINGLE

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

MODEL_MAPPING = {
    'CNN_LSTM_ATT_SINGLE': CNN_LSTM_ATT_SINGLE,
    'BSP': BSP,
}



def batch_spectral_penalization(feature_source, feature_target, k=1):
    _, singular_source, _ = feature_source.svd()
    _, singular_target, _ = feature_target.svd()
    bsp_loss = torch.sum(singular_source[:k] ** 2 +
                         singular_target[:k] ** 2)
    return bsp_loss


class ModelTrainer(nn.Module):

    def __init__(self, lr, test_step, lr_decay, feature_type, model, **kwargs):
        super(ModelTrainer, self).__init__()
        # 定义模型
        # model_gpu = nn.DataParallel(MODEL_MAPPING[model]())
        model_gpu = MODEL_MAPPING[model]()
        self.model = model_gpu.to(device)
        # 定义损失函数
        self.src_task_loss = nn.CrossEntropyLoss().to(device)
        self.domain_loss = nn.CrossEntropyLoss().to(device)
        # 定义优化器
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        # 打印模型参数大小
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in parameters) / 1024 / 1024))

    def train_network(self, epoch, train_loader, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        lr = self.optim.param_groups[0]['lr']
        total_loss = 0
        total_src_domain_loss = 0
        total_target_domain_loss = 0
        total_src_task_loss = 0
        torch.cuda.empty_cache()
        start_steps = (epoch - 1) * len(train_loader)
        total_steps = 30 * len(train_loader)

        train_iter = iter(train_loader)
        loader_iter = iter(loader)
        # 获取两个加载器的长度
        len_train = len(train_loader)
        len_loader = len(loader)

        # 计算需要重复多少次 loader 才能匹配 train_loader 的长度
        repeats = (len_train // len_loader) + 1 if len_train % len_loader != 0 else len_train // len_loader

        # 创建一个扩展的 loader 迭代器
        extended_loader = itertools.chain.from_iterable(itertools.repeat(loader_iter, repeats))

        # 遍历 train_loader 和扩展的 loader
        for num, ((src_data, src_labels), (target_data)) in enumerate(zip(train_iter, extended_loader), start=1):
            self.optim.zero_grad()
            # p = float(num - 1 + start_steps) / total_steps
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 0.5
            # print(p, alpha)
            src_labels = torch.LongTensor(src_labels)
            src_labels = src_labels.to(device)

            src_domain_labels = torch.LongTensor(torch.zeros((len(src_data)), dtype=torch.int64))
            src_domain_labels = src_domain_labels.to(device)

            src_data = src_data.to(device)

            target_data = target_data.to(device)
            target_domain_labels = torch.LongTensor(torch.ones((len(target_data)), dtype=torch.int64))
            target_domain_labels = target_domain_labels.to(device)

            src_task_predict, src_domain_predict, feature_source = self.model.forward(src_data, alpha)
            _, target_domain_predict, feature_target = self.model.forward(target_data, alpha)
            bsp_loss = batch_spectral_penalization(feature_source, feature_target, 1)

            src_task_loss = self.src_task_loss.forward(src_task_predict, src_labels)

            src_domain_loss = self.domain_loss.forward(src_domain_predict, src_domain_labels)
            target_domain_loss = self.domain_loss.forward(target_domain_predict,
                                                          target_domain_labels)
            losses = src_task_loss + 1 * (src_domain_loss + target_domain_loss) + bsp_loss * 1e-2

            losses.backward()
            self.optim.step()

            total_loss += losses
            total_src_task_loss += src_task_loss
            total_src_domain_loss += src_domain_loss
            total_target_domain_loss += target_domain_loss
            # 清空显存缓存
            torch.cuda.empty_cache()
        print(total_src_task_loss)
        print(total_src_domain_loss)
        print(total_target_domain_loss)
        torch.cuda.empty_cache()
        return lr, total_loss

    def eval_network(self, devel_loader):
        self.eval()
        # label to emotion
        l2e = {v: k for k, v in label_MAPPING.items()}

        # 预测结果，实际结果
        pred_list, true_list, pred_num = [], [], []
        with torch.no_grad():
            torch.cuda.empty_cache()
            for num, (data, labels) in enumerate(devel_loader, start=1):
                labels = torch.LongTensor(labels)
                labels = labels.to(device)
                data = data.to(device)
                # output = self.model.forward(data)
                output, _, _ = self.model.forward(data, 0)
                # 清空显存缓存
                torch.cuda.empty_cache()
                result_list = output.argmax(dim=1).tolist()
                pred_num.extend(result_list)
                pred_list.extend(list(map(lambda x: l2e[x], result_list)))
                true_list.extend(list(map(lambda x: l2e[x], labels.cpu().numpy())))
        # 计算UAR(未加权平均召回)
        uar = float(recall_score(
            true_list,
            pred_list,
            average="macro",
        ))
        torch.cuda.empty_cache()
        return uar, true_list, pred_list, pred_num

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
