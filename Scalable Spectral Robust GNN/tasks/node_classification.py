import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tasks.base_task import BaseTask
from sklearn.model_selection import train_test_split
from tasks.utils import accuracy, node_cls_train, node_cls_mini_batch_train, node_cls_evaluate, node_cls_mini_batch_evaluate
from sklearn.manifold import TSNE

class NodeClassification(BaseTask):
    def __init__(self, dataset, model, normalize_times,
                 lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(),
                 train_batch_size=None, eval_batch_size=None):
        super(NodeClassification, self).__init__()
        self.normalize_times = normalize_times
        self.normalize_record = {"val_acc": [], "test_acc": []}

        self.dataset = dataset
        self.labels = self.dataset.y

        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.device = device
        #self.log = './SIGN_log.txt'
        self.mini_batch = False
        if train_batch_size is not None:
            self.mini_batch = True
            print(f"Mini-batch training size: {train_batch_size}, eval and test size: {eval_batch_size}")
            self.train_loader = DataLoader(
                self.dataset.train_idx, batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.val_loader = DataLoader(
                self.dataset.val_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.test_loader = DataLoader(
                self.dataset.test_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.all_eval_loader = DataLoader(
                range(self.dataset.num_node), batch_size=eval_batch_size, shuffle=False, drop_last=False)


        for i in range(self.normalize_times):
            if i == 0:
                normalize_times_st = time.time()
            self.execute()
        
        if self.normalize_times > 1:
            print("Optimization Finished!")
            print("Total training time is: {:.4f}s".format(time.time() - normalize_times_st))
            print("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(self.normalize_record["val_acc"]), 4), round(np.std(self.normalize_record["val_acc"], ddof=1), 4), round(np.mean(self.normalize_record["test_acc"]), 4), round(np.std(self.normalize_record["test_acc"], ddof=1), 4)))

        

    def get_test_acc(self):
        return np.mean(self.normalize_record["test_acc"])

    def execute(self):
        pre_time_st = time.time()
        self.model.preprocess(self.dataset.adj , self.dataset.x)
        pre_time_ed = time.time()
        
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        loss = []
        for epoch in range(self.epochs):
            t = time.time()
            if self.mini_batch is False:
                loss_train, acc_train = node_cls_train(self.model, self.dataset.train_idx, self.labels, self.device,
                                              self.optimizer, self.loss_fn)
                acc_val, acc_test = node_cls_evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                             self.labels, self.device)
            else:
                loss_train, acc_train = node_cls_mini_batch_train(self.model, self.dataset.train_idx, self.train_loader,
                                                         self.labels, self.device, self.optimizer, self.loss_fn)
                acc_val, acc_test = node_cls_mini_batch_evaluate(self.model, self.dataset.val_idx, self.val_loader,
                                                        self.dataset.test_idx, self.test_loader, self.labels,
                                                        self.device)
            loss.append(loss_train)
            if self.normalize_times == 1:
                print("Epoch: {:03d}, loss_train: {:.4f}, acc_train: {:.4f}, acc_val: {:.4f},"
                                 "acc_test: {:.4f}".format(epoch+1, loss_train, acc_train, acc_val, acc_test))

            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test

        acc_val, acc_test = self.postprocess()
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test
        # tsne = TSNE(n_components=2)
        # x_tsne = tsne.fit_transform(self.model.model_forward(range(self.dataset.num_node), self.device).cpu().detach().numpy())
        # y = self.dataset.y.cpu().numpy()
        # plt.figure(figsize=(10, 10))
        # plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
        # plt.colorbar(ticks=range(10))
        # plt.clim(-0.5, 9.5)
        # plt.savefig('test.png')
        # plt.show()

        #损失函数变化曲线
        # plt.figure(figsize=(10, 10))
        # plt.plot(loss)
        # plt.savefig('loss.png')
        if self.normalize_times == 1:
            print("Optimization Finished!")
            print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
            # file = open(self.log, 'a')
            # file.write(f'{best_val:.4f}  {best_test:.4f}\n')
            # file.close()
        self.normalize_record["val_acc"].append(best_val)
        self.normalize_record["test_acc"].append(best_test)

    def postprocess(self):
        self.model.eval()
        if self.model.post_graph_op is not None:
            if self.mini_batch is False:
                outputs = self.model.model_forward(
                    range(self.dataset.num_node), self.device)
            else:
                outputs = None
                for batch in self.all_eval_loader:
                    output = self.model.model_forward(batch, self.device)
                    if outputs is None:
                        outputs = output
                    else:
                        outputs = torch.vstack((outputs, output))
            final_output = self.model.postprocess(self.dataset.adj, outputs)
            acc_val = accuracy(final_output[self.dataset.val_idx], self.labels[self.dataset.val_idx])
            acc_test = accuracy(final_output[self.dataset.test_idx], self.labels[self.dataset.test_idx])
        else:
            if self.mini_batch is False:
                acc_val, acc_test = node_cls_evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                             self.labels, self.device)
            else:
                acc_val, acc_test = node_cls_mini_batch_evaluate(self.model, self.dataset.val_idx, self.val_loader,
                                                        self.dataset.test_idx, self.test_loader, self.labels,
                                                        self.device)
        return acc_val, acc_test


