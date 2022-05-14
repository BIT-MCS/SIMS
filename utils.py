import numpy as np
import time
import os
import torch.nn as nn
import torch


def init(module, weight_init, gain=1, name=None):
    if name is not None and 'lstmcell' in name:
        weight_init(module.weight_hh, gain=gain)
        weight_init(module.weight_ih, gain=gain)
        nn.init.constant_(module.bias_hh, 0)
        nn.init.constant_(module.bias_ih, 0)
        return module
    weight_init(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module


class LogUtil():
    def __init__(self, root_path, method_name, is_train=True):
        self.time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
        self.is_train = is_train
        if is_train:
            self.full_path = os.path.join(root_path, method_name, self.time)

        else:
            self.full_path = root_path
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

        if self.is_train:
            self.loss_path = os.path.join(self.full_path, 'loss.npy')
            self.loss_list = []

            self.report_path = self.full_path + '/train_log.txt'

            self.model_path = self.full_path + '/model.pth'

    def record_loss(self, loss):
        self.loss_list.append(loss)
        np.save(self.loss_path, np.asarray(self.loss_list))

    def record_report(self, report_str):
        f = open(self.report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        print('model has been save to ', self.model_path)
