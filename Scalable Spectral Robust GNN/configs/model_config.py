import argparse

parser = argparse.ArgumentParser()

# simplex graph
#   (baseline) sgc gamlp gcn gbp nafs sign wavelet
parser.add_argument('--model_name', help='gnn model', type=str, default="wavelet")
parser.add_argument('--num_layers', help='number of gnn layers', type=int, default=3)
parser.add_argument('--dropout', help='drop out of gnn model', type=float, default=0.5)
parser.add_argument('--hidden_dim', help='hidden units of gnn model', type=int, default=256)
# scalable gnn model
parser.add_argument('--prop_steps', help='prop steps', type=int, default=3)
# adj normalize
parser.add_argument('--r', help='symmetric normalized unit', type=float, default=0.5)
parser.add_argument('--ppr_alpha', help='ppr approxmite symmetric adj unit', type=float, default=0.1)
parser.add_argument('--message_alpha', help='weighted message operator', type=float, default=0.5)
parser.add_argument('--q', help='the imaginary part of the complex unit', type=float, default=0.05)
# wavelet
parser.add_argument("--approximation-order", type=int, default=3, help="Order of Chebyshev polynomial. Default is 3.")
parser.add_argument("--step_size", type=int, default=20, help="Number of steps. Default is 20.")
parser.add_argument("--switch", type=int, default=100, help="Number of dimensions. Default is 100.")
parser.add_argument("--tolerance", type = float, default=10**-4, help="Sparsification parameter. Default is 10^-4.")
parser.add_argument("--scale", type=float, default=0.5, help="Heat kernel scale length. Default is 1.0.")

model_args = parser.parse_args()

