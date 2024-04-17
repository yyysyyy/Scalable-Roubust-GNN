from models.gcn import GCN
from models.sgc import SGC 
from models.gbp import GBP 
from models.sign import SIGN
from models.nafs import NAFS
from models.ssgc import SSGC 
from models.gamlp import GAMLP  
from models.walvet import WAVELAT
from models.clean_train_model import CleanTrainModel
from configs.model_config import model_args

def load_model(feat_dim, output_dim, ncount):
    
    if model_args.model_name == "gcn":
        print(f"model: {model_args.model_name}, r: {model_args.r}, hidden_dim: {model_args.hidden_dim}, num_layers: {model_args.num_layers}, dropout: {model_args.dropout}")
        model = GCN(r=model_args.r, feat_dim=feat_dim, hidden_dim=model_args.hidden_dim, output_dim=output_dim, dropout=model_args.dropout)

    elif model_args.model_name == "sgc":
        print(f"model: {model_args.model_name}, prop_steps: {model_args.prop_steps}, r: {model_args.r}")
        model = SGC(prop_steps=model_args.prop_steps, r=model_args.r, feat_dim=feat_dim, output_dim=output_dim)

    elif model_args.model_name == "ssgc":
        print(f"model: {model_args.model_name}, prop_steps: {model_args.prop_steps}, r: {model_args.r}")
        model = SSGC(prop_steps=model_args.prop_steps, r=model_args.r, feat_dim=feat_dim, output_dim=output_dim)

    elif model_args.model_name == "nafs":
        print(f"model: {model_args.model_name}, prop_steps: {model_args.prop_steps}, r: {model_args.r}")
        model = NAFS(prop_steps=model_args.prop_steps, r=model_args.r, feat_dim=feat_dim, output_dim=output_dim)

    elif model_args.model_name == "sign":
        print(f"model: {model_args.model_name}, prop_steps: {model_args.prop_steps}, r: {model_args.r}, hidden_dim: {model_args.hidden_dim}, num_layers: {model_args.num_layers}, dropout: {model_args.dropout}")
        model = SIGN(prop_steps=model_args.prop_steps, r=model_args.r, 
                        feat_dim=feat_dim, hidden_dim=model_args.hidden_dim, output_dim=output_dim,
                        num_layers=model_args.num_layers, dropout=model_args.dropout)

    elif model_args.model_name == "gbp":
        print(f"model: {model_args.model_name}, prop_steps: {model_args.prop_steps}, r: {model_args.r}, hidden_dim: {model_args.hidden_dim}, num_layers: {model_args.num_layers}, dropout: {model_args.dropout}, message_alpha: {model_args.message_alpha}")
        model = GBP(prop_steps=model_args.prop_steps, r=model_args.r, message_alpha=model_args.message_alpha,
                       feat_dim=feat_dim, hidden_dim=model_args.hidden_dim, output_dim=output_dim,
                       num_layers=model_args.num_layers, dropout=model_args.dropout)

    elif model_args.model_name == "gamlp":
        print(f"model: {model_args.model_name}, prop_steps: {model_args.prop_steps}, r: {model_args.r}, hidden_dim: {model_args.hidden_dim}, num_layers: {model_args.num_layers}, dropout: {model_args.dropout}")
        model = GAMLP(prop_steps=model_args.prop_steps, r=model_args.r, 
            feat_dim=feat_dim, hidden_dim=model_args.hidden_dim, output_dim=output_dim,
            num_layers=model_args.num_layers, dropout=model_args.dropout)
    elif model_args.model_name == "wavelet":
        print(f"model: {model_args.model_name}, scale: {model_args.scale}, approximation_order: {model_args.approximation_order}, tolerance: {model_args.tolerance}, hidden_dim: {model_args.hidden_dim}, dropout: {model_args.dropout}")
        model = WAVELAT(ncount= ncount,scale=model_args.scale, approximation_order=model_args.approximation_order, tolerance=model_args.tolerance,feat_dim=feat_dim, hidden_dim=model_args.hidden_dim, output_dim=output_dim, dropout=model_args.dropout)
    else:
        return NotImplementedError
    return model