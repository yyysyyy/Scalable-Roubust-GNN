import time
import torch
import datetime

from models.model_init import load_model
from configs.data_config import data_args
from utils import seed_everything, get_params
from configs.training_config import training_args
from tasks.node_classification import NodeClassification    
from datasets.simhomo.load_homo_simplex_real_data import load_homo_simplex_dataset
from sparsity_datasets.simhomo.load_homo_simplex_real_sparsity_data import load_homo_simplex_sparsity_dataset
from configs.model_config import model_args


if __name__ == "__main__":
    run_id = f"{time.time():.8f}"
    print(f"program start: {datetime.datetime.now()}")

    # set up seed
    seed_everything(training_args.seed)
    device = torch.device('cuda:{}'.format(training_args.gpu_id) if (training_args.use_cuda and torch.cuda.is_available()) else 'cpu')
    print(f"Load homogeneous simplex network: {data_args.data_name}")
    set_up_datasets_start_time = time.time()
    dataset = load_homo_simplex_sparsity_dataset(name=data_args.data_name, root=data_args.data_root, split=data_args.data_split, is_augumented=True)
    set_up_datasets_end_time = time.time()
    print(f"datasets: {data_args.data_name}, root dir: {data_args.data_root}, node-level split method: {data_args.data_split}, the running time is: {round(set_up_datasets_end_time-set_up_datasets_start_time,4)}s")

    model = load_model(feat_dim=dataset.num_features, output_dim=dataset.num_classes, ncount = dataset.x.shape[0])
    NodeClassification(dataset, model, normalize_times=training_args.normalize_times, lr=training_args.lr, weight_decay=training_args.weight_decay, epochs=training_args.num_epochs , device=device)  