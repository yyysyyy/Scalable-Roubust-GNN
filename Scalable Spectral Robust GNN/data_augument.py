import time
import torch
import datetime
import os
import pickle as pkl
import numpy as np
import random
from rich.progress import Progress
from utils import seed_everything, generate_numbers, compute_distance
from configs.training_config import training_args  
from tasks.model_train import TrainModel
from models.clean_train_model import CleanTrainModel
from models.base_scalable.simple_models import FeatureAugument2MLP
from sparsity_datasets.simhomo.load_homo_simplex_real_sparsity_data import load_homo_simplex_sparsity_dataset
from configs.data_augument_config import data_augument_args
from tasks.utils import accuracy
from collections import Counter

def feature_augument(dataset, data_augument_model, lr, weight_decay, epochs, device, batch_size):

    optimizer = torch.optim.Adam(data_augument_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn1 = torch.nn.L1Loss()
    loss_fn2 = torch.nn.CrossEntropyLoss()
    train_idx, val_idx, test_idx = dataset.train_idx, dataset.val_idx, dataset.test_idx
    adj = dataset.adj.tocoo()
    row, col = torch.tensor(adj.row, dtype=torch.long), torch.tensor(adj.col, dtype=torch.long)
    data_augument_model.to(device)
    sparsity_x = torch.from_numpy(dataset.x * dataset.feature_mask.numpy().astype(np.float32)).to(device)
    x = torch.from_numpy(dataset.x).to(device)
    best_acc = 0
    best_model = None
    with Progress() as progress:
        task = progress.add_task("[cyan]Progress:", total=epochs)
        for epoch in range(epochs): 
            data_augument_model.train()
            optimizer.zero_grad()
            sp_mid_dim, sp_output = data_augument_model(sparsity_x)
            mid_dim, output = data_augument_model(x)
            loss1 = loss_fn1(sp_output[train_idx], output[train_idx])
            loss2 = loss_fn2(output[train_idx], dataset.y[train_idx].to(device))
            loss3 = loss_fn2(sp_output[train_idx], dataset.y[train_idx].to(device))
            #row_output, col_output = sp_output[row], sp_output[col]
            #loss4 =  torch.cdist(row_output, col_output).mean()
            loss =  loss2 
            #loss = 0.1 * loss_fn2(sp_output, dataset.y[dataset.train_idx].to(device)) + 0.1 * loss_fn2(output, dataset.y[dataset.train_idx].to(device)) #39.7 64.8 45.4
            #loss = 0.1 * loss_fn2(output, dataset.y[dataset.train_idx].to(device))  #43 66.5   44.3
            loss.backward()
            optimizer.step()
            data_augument_model.eval()
            _, test_output = data_augument_model(sparsity_x[test_idx])
            _, test_output2 = data_augument_model(x[test_idx])
            test_acc = accuracy(test_output, dataset.y[test_idx].to(device))
            test_acc2 = accuracy(test_output2, dataset.y[test_idx].to(device))
            #print("Epoch: {:03d}, loss: {:.4f}, test_acc: {:.4f}, test2_acc: {:.4f}".format(epoch+1, loss.item(), test_acc, test_acc2))
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = data_augument_model
            #加一个进度条,只有进度条就行了
            progress.update(task, completed=epoch+1)
            progress.refresh()
        
    print("best_acc: {:.4f}".format(best_acc))
    feature = torch.zeros(dataset.x.shape[0], data_augument_args.hidden_dim + dataset.num_classes).to(device)
    data_augument_model.eval()
    #mid_dim, soft_label = best_model(torch.from_numpy(dataset.x).to(device))
    mid_dim, soft_label = best_model(sparsity_x)
    soft_label = torch.nn.functional.softmax(soft_label, dim=1)
    feature = torch.cat([mid_dim, soft_label], dim=1)
    feature = feature.detach()
    return feature, soft_label


def edge_augument(dataset, soft_label):
    edge_row, edge_col = dataset.edge.row, dataset.edge.col
    edge_index = torch.cat((edge_row.unsqueeze(0), edge_col.unsqueeze(0)), dim=0)
    edge_total = torch.cat((edge_row, edge_col),dim = 0)
    edge_total = edge_total.numpy()
    counts = Counter(edge_total)
    for i in range(dataset.x.shape[0]):
        if i not in counts:
            counts.update({i:0})
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
    numbers = list(range(0,soft_label.shape[0]))
    for node, degree in sorted_counts:
        if degree >= data_augument_args.degree_level:
            break
        else:
            candiate_nodes = generate_numbers((data_augument_args.degree_level - degree) * 100, node, numbers)
            candiate_soft_label = soft_label[candiate_nodes]
            node_soft_label = soft_label[node]
            distance = compute_distance(candiate_soft_label, node_soft_label)
            _ , sorted_indices = torch.sort(distance, dim = 0)
            sorted_indices = sorted_indices.tolist()
            sorted_indices = [candiate_nodes[i] for i in sorted_indices]
            node_pairs = torch.zeros((2, data_augument_args.degree_level - degree), dtype = torch.long)
            node_pairs[0,:] = node
            node_pairs[1,:] = torch.tensor(sorted_indices[:data_augument_args.degree_level - degree])
            edge_index = torch.cat((edge_index, node_pairs), dim = 1)
    row, col = edge_index[0,:], edge_index[1,:]
    edge_index2 = torch.cat((col.unsqueeze(0), row.unsqueeze(0)), dim=0)
    edge_index = torch.cat((edge_index, edge_index2), dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index
        
        

def data_save(feature, edge_index, dataset):
    path = data_augument_args.data_save_path + data_augument_args.data_name + '/raw/'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    feature_file = path + 'feature.pt'
    edge_index_file = path + 'edge_index.pt'
    label_file = path + 'label.pt'
    train_idx_file = path + 'train_idx.pt'
    val_idx_file = path + 'val_idx.pt'
    test_idx_file = path + 'test_idx.pt'
    feature_mask_file = path + 'feature_mask.pt'
    edge_mask_file = path + 'edge_mask.pt'
    with open(feature_file, 'wb') as f:
        torch.save(feature, f)
    with open(edge_index_file, 'wb') as f:
        # adj = dataset.adj.tocoo()
        # row, col = torch.tensor(adj.row, dtype=torch.long), torch.tensor(adj.col, dtype=torch.long)
        # edge_index = torch.stack([row, col], dim=0)
        torch.save(edge_index, f)
    with open(label_file, 'wb') as f:
        torch.save(dataset.y, f)
    with open(train_idx_file, 'wb') as f:
        torch.save(dataset.train_idx, f)
    with open(val_idx_file, 'wb') as f:
        torch.save(dataset.val_idx, f)
    with open(test_idx_file, 'wb') as f:
        torch.save(dataset.test_idx, f)
    with open(feature_mask_file, 'wb') as f:
        torch.save(dataset.feature_mask, f)
    with open(edge_mask_file, 'wb') as f:
        torch.save(dataset.edge_mask, f)

if __name__ == "__main__":
    run_id = f"{time.time():.8f}"
    print(f"program start: {datetime.datetime.now()}")

    # set up seed
    seed_everything(training_args.seed)
    device = torch.device('cuda:{}'.format(training_args.gpu_id) if (training_args.use_cuda and torch.cuda.is_available()) else 'cpu')

    print(f"Load homogeneous simplex network: {data_augument_args.data_name}")
    set_up_datasets_start_time = time.time()
    dataset = load_homo_simplex_sparsity_dataset(name=data_augument_args.data_name, root=data_augument_args.data_root, split=data_augument_args.data_split)
    set_up_datasets_end_time = time.time()
    print(f"datasets: {data_augument_args.data_name}, root dir: {data_augument_args.data_root}, node-level split method: {data_augument_args.data_split}, the running time is: {round(set_up_datasets_end_time-set_up_datasets_start_time,4)}s")   
    data_augument_model = FeatureAugument2MLP(dataset.num_features, data_augument_args.hidden_dim, dataset.num_classes, data_augument_args.dropout)
    feature, soft_label = feature_augument(dataset, data_augument_model, lr=0.01, weight_decay=data_augument_args.weight_decay, epochs=200 , device=device, batch_size = data_augument_args.batch_size)
    edge_index = edge_augument(dataset, feature)
    data_save(feature, edge_index,dataset)
    print('done')