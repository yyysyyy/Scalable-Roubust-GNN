import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import argparse
import time
import copy
from torch_geometric.loader import NeighborSampler
from torch_geometric.loader import ClusterLoader, ClusterData
from model import MLP, GCN, SAGE, GAT, SGC, SIGN
from logger import Logger
from ogb.nodeproppred import Evaluator
from utils import dataRead, sgc_precompute

def train(model, data, train_idx, optimizer, args, device):
    data = data.to(device)
    model.train()
    optimizer.zero_grad()
    if args.model != 'MLP':
        out = model(data.x, data.adj_t)[train_idx]
    else:
        out = model(data.x)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx].view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

def train_minibatch(model, data, train_idx, optimizer, args):
    cluster_data = ClusterData(data, num_parts=128)
    train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    for sub_data in train_loader:
        sub_data = sub_data.to(args.device)
        if args.model != 'MLP':
            out = model(sub_data.x, sub_data.adj_t)[sub_data.train_mask]
        else:
            out = model(sub_data.x)[sub_data.train_mask]
        loss = F.nll_loss(out, sub_data.y[sub_data.train_mask].view(-1))
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss


@torch.no_grad()
def test(model, data, evaluator, args):
    model.eval()
    if args.model != 'MLP':
        out = model(data.x, data.adj_t)
    else:
        out = model(data.x)
    y_pred = out.argmax(dim=1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask].view(-1, 1),
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_mask].view(-1, 1),
        'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask].view(-1, 1),
        'y_pred': y_pred[data.test_mask],
    })['acc']

    return train_acc, valid_acc, test_acc

def choose_model(args):
    if args.model == 'MLP':
        model = MLP(data.num_features, args.hidden, data.y.max().item()+1, args.layers,args.dropout)
    elif args.model == 'GCN':
        model = GCN(data.num_features, args.hidden, data.y.max().item()+1, args.layers,args.dropout)
    elif args.model == 'SAGE':
        model = SAGE(data.num_features, args.hidden, data.y.max().item()+1, args.layers,args.dropout)
    elif args.model == 'GAT':
        model = GAT(data.num_features, args.hidden, data.y.max().item()+1, args.layers,args.dropout, 8)
    elif args.model == 'SGC':
        model = SGC(data.num_features, data.y.max().item()+1)
    elif args.model == 'SIGN':
        model = SIGN(data.num_features, data.y.max().item()+1)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True,help='Use CUDA training.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu rank')
    parser.add_argument('--batch_size', type=int, default=32, help='Min bath size.')
    parser.add_argument('--epochs', type=int, default=400,help='Number of epochs to train.')
    parser.add_argument('--model', type=str, default='SGC', help= 'MLP, GCN, GAT, SAGE, SGC, SIGN, JKNet, APPNP, GBP, GAMLP')
    parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
    parser.add_argument('--samples', type=int, default=10000,
                        help='samples per triplet loss.')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataroot', type=str, default='./dataset', help='path')
    parser.add_argument('--layers', type=int, default=2, help='num of layers')
    parser.add_argument('--runs', type=int, default=3, help='num of runs')
    parser.add_argument('--dataset', type=str, default='cora', 
                        help='cora, citeseer, pubmed, reddit, flickr, ogbn-arxiv, ogbn-products, ogbn-papers100M')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    data = dataRead(args.dataroot, args.dataset)
    logger = Logger(args.runs)
    model = choose_model(args).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    if args.dataset == 'ogbn-arxiv' or args.dataset == 'ogbn-products' or args.dataset == 'ogbn-papers100M':
        evaluator = Evaluator(name=args.dataset)
    else:
        evaluator = Evaluator(name='ogbn-arxiv')
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, data.train_mask, optimizer, args, device)
            result = test(model, data, evaluator, args)
            logger.add_result(run, result)
            if epoch % 1 == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

