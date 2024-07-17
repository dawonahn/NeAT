import pdb
# import wandb
import argparse

from dotmap import DotMap
from read import *
from train import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--rpath', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, help='Dataset name')

    # Model
    parser.add_argument('--tf', type=str, help='cp, neat', default='neat')
    parser.add_argument('--lossf', type=str, help='MSELoss or BCELoss', default='BCELoss')

    # Hyper-params for tf
    parser.add_argument('--rank', type=int, help='Rank size of tf')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--bs', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Epochs for training')
    
    # Hyper-params for nn model
    parser.add_argument('--layer_dims', type=str, help='Layer dimensions')
    parser.add_argument('--dropout', type=float, help='Dropout on rank-1 component')
    parser.add_argument('--dropout2', type=float, help='Dropout inside network')

    # Etc
    parser.add_argument('--device', type=str, help='Device: cuda:0')

    args = parser.parse_args()
    dict_args = DotMap(dict(args._get_kwargs()))
    if dict_args.layer_dims is not None:
        dict_args.layer_dims = [int(dims) for dims in dict_args.layer_dims.split(',')]
        dict_args.depth = len(dict_args.layer_dims) - 1
    else:
        dict_args.depth = 0
    print(dict_args)
    return dict_args

def main():

    cfg = parse_args()

    # run = wandb.init(
    #         project='NeAT',
    #         group=cfg.dataset,
    #         job_type=str(cfg.rank),
    #         config = cfg,
    # )

    tensor = read_data(cfg)
    model = train(tensor, cfg, wandb)

    if wandb is not None:
        cfg.wnb_name = wandb.run.name
        cfg.model_path = save_checkpoints(model, cfg)
        wandb.config.update(cfg, allow_val_change=True)

if __name__ ==  '__main__':
    main()


