import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tasks.base_task import BaseTask
from tasks.utils import accuracy, node_cls_train, node_cls_mini_batch_train, node_cls_evaluate, node_cls_mini_batch_evaluate


class TrainModel(BaseTask):
    def __init__(self, dataset, model, normalize_times,
                 lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(),
                 train_batch_size=None, eval_batch_size=None):
        super(TrainModel, self).__init__()
        self.normalize_times = normalize_times
        self.normalize_record = {"val_acc": [], "test_acc": []}

        self.dataset = dataset
        self.labels = self.dataset.y
        self.train_idx = self.dataset.train_idx
        self.val_idx = self.dataset.val_idx
        self.test_idx = self.dataset.test_idx
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.device = device
        self.mini_batch = False
        if train_batch_size is not None:
            self.mini_batch = True
            print(f"Mini-batch training size: {train_batch_size}, eval and test size: {eval_batch_size}")
            self.train_loader = DataLoader(
                self.dataset.train_idx, batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.all_eval_loader = DataLoader(
                range(self.dataset.num_node), batch_size=eval_batch_size, shuffle=False, drop_last=False)
        self.execute()
    def execute(self):
        pre_time_st = time.time()
        self.model.preprocess(self.dataset.adj , self.dataset.x)
        #self.model.processed_feature = torch.from_numpy(self.dataset.x).float()
        pre_time_ed = time.time()
        
        if self.normalize_times == 1:
            print(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)

        t_total = time.time()
        for epoch in range(self.epochs):
            t = time.time()
            if self.mini_batch is False:
                self.model.train()
                self.optimizer.zero_grad()
                _, train_output = self.model.model_forward(self.train_idx, self.device)
                loss_train = self.loss_fn(train_output, self.labels[self.train_idx])
                acc_train = accuracy(train_output, self.labels[self.train_idx])
                loss_train.backward()
                self.optimizer.step()
                loss_train, acc_train = loss_train.item(), acc_train
                self.model.eval()
                _,val_output = self.model.model_forward(self.val_idx, self.device)
                _,test_output = self.model.model_forward(self.test_idx, self.device)
                acc_val = accuracy(val_output, self.labels[self.val_idx])
                acc_test = accuracy(test_output, self.labels[self.test_idx])
                print(" acc_val: {:.4f}, acc_test: {:.4f}, time: {:.4f}s".format(acc_val, acc_test, time.time() - t))
            else:
                self.model.train()
                correct_num = 0
                loss_train_sum = 0.
                for batch in self.train_loader:
                    _, train_output = self.model.model_forward(batch, self.device)
                    loss_train = self.loss_fn(train_output, self.labels[batch])
                    pred = train_output.max(1)[1].type_as(self.labels)
                    correct_num += pred.eq(self.labels[batch]).double().sum()
                    loss_train_sum += loss_train.item()
                    self.optimizer.zero_grad()
                    loss_train.backward()
                    self.optimizer.step()
                loss_train = loss_train_sum / len(self.train_loader)
                acc_train = correct_num / len(self.train_idx)
            if self.normalize_times == 1:
                print("Epoch: {:03d}, loss_train: {:.4f}, acc_train: {:.4f}, time: {:.4f}s".format(epoch+1, loss_train, acc_train, time.time() - t))
            
            # if acc_val > best_val:
            #     best_val = acc_val
            #     best_test = acc_test

        # acc_val, acc_test = self.postprocess()
        # if acc_val > best_val:
        #     best_val = acc_val
        #     best_test = acc_test

        if self.normalize_times == 1:
            print("Optimization Finished!")
            print("Total training time is: {:.4f}s".format(time.time() - t_total))
    
    def get_mid_dim(self, batch_x):
        
        mid, _ = self.model.model_forward(batch_x, self.device)
        return mid
        