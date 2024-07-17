
import pdb
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from read import *
from model import *
from utils import *
from metrics import *


def train(tensor, cfg, wandb=None, verbose=True):

    print("Start training a neural additive tensor decomposition...!")
    # Prepare dataset
    dataset = COODataset(tensor.train_i, tensor.train_v)
    dataloader = DataLoader(dataset, batch_size=cfg.bs, shuffle=True)

    # create the model
    m = nn.Sigmoid()
    if cfg.lossf == 'BCELoss':
        loss_fn = nn.BCELoss(reduction='mean')
    else:
        loss_fn = nn.MSELoss()

    model = NeAT(cfg, tensor.sizes).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    flag=0
    old_valid_rmse = 1e+6
    best_valid_rmse = 1e+6
    total_running_time = 0
    start = time.time()
    for epoch in range(cfg.epochs):

        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch[0], batch[1]
            outputs = model(inputs)
            loss = loss_fn(m(outputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_end = time.time()
        total_running_time += (epoch_end - epoch_start)

        model.eval()
        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                val_rec = m(model(tensor.valid_i))
                train_rmse = np.sqrt(epoch_loss/len(dataloader))
                valid_rmse = rmse(val_rec, tensor.valid_v)

                if wandb:
                    wandb.log({"train_rmse":train_rmse,
                               "valid_rmse":valid_rmse,
                               "total_running_time":total_running_time})
                    
                if verbose:
                    print(f"Epochs {epoch} TrainRMSE: {train_rmse:.4f}\t"
                            f"ValidRMSE: {valid_rmse:.4f}\t")

                if (old_valid_rmse <= valid_rmse):
                    flag +=1
                old_valid_rmse = valid_rmse
                
                if flag == 5:
                    break

    training_time = time.time() - start
    model.eval()
    with torch.no_grad(): 
        test_rec =  m(model(tensor.test_i))
        if tensor.bin_val:
            r = eval_(test_rec.data, tensor.test_v)
            r['test_rmse'] = rmse(test_rec, tensor.test_v)
            if verbose:
                print(f"TestRMSE:{r['test_rmse']:.4f} Acc:{r['acc']:.4f} Recall:{r['recall']:.4f}"
                    f"Prec.:{r['prec']:.4f} F1:{r['f1']:.4f} AUC : {r['auc']:.4f}")
        else:
            r={'test_rmse':rmse(test_rec, tensor.test_v)}
        if wandb:
            r['avg_running_time'] = total_running_time / epoch
            r['all_total_training_time'] = training_time
            wandb.log(r)

    model.result = r
    return model
