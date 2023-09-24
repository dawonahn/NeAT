
import torch
import torchmetrics.functional as tmf


def rmse(pred, true):
    return torch.sqrt(torch.mean(torch.square(pred - true)))

def nre(pred, true):
    diff1 = torch.sqrt(torch.sum(torch.square(pred - true)))
    diff2 = torch.sqrt(torch.sum(torch.square(true)))
    return diff1/diff2

def eval_(pred, true, q=0.5, task='binary'):
    acc = tmf.accuracy(pred>q, true, task=task)
    recall = tmf.recall(pred>q, true, task=task)
    prec = tmf.precision(pred>q, true, task =task)
    f1 = tmf.f1_score(pred, true, task=task)
    auc = tmf.auroc(pred, true, task=task)
        
    return {"acc" :acc.item(), "recall":recall.item(),
            "prec":prec.item(),"f1":f1.item(), "auc":auc.item()}
        