import argparse

parser = argparse.ArgumentParser()


# seed
parser.add_argument('--seed', help='seed everything', type=int, default=2023)
# use gpu
parser.add_argument('--use_cuda', help='use gpu', type=bool, default=True)
parser.add_argument('--gpu_id', help='gpu id', type=int, default=0)
parser.add_argument('--normalize_times', help='normalize train', type=int, default=1)
# training details
# parser.add_argument('--gnn_batch_size', help='batch size', type=int, default=64)
parser.add_argument('--num_epochs', help='number of epochs', type=int, default=300)
parser.add_argument('--lr', help='learning rate of gnn model', type=float, default=0.001)
parser.add_argument('--weight_decay', help='weight decay of gnn model', type=float, default=10**-5)
parser.add_argument('--batch_train', help='training batch size', type=int, default=64)
parser.add_argument('--train_batch_size', help='training batch size', type=int, default=100)
parser.add_argument('--eval_batch_size', help='val and test batch size', type=int, default=100)


training_args = parser.parse_args()

