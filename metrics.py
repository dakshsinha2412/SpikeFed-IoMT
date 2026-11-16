"""Dependency-light classification metrics for evaluated predictions."""
from __future__ import annotations
import torch
def classification_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    cm=torch.zeros(num_classes,num_classes,dtype=torch.int64)
    for t,p in zip(target.flatten().cpu(),pred.flatten().cpu()): cm[t,p]+=1
    tp=cm.diag().float(); precision=tp/cm.sum(0).clamp_min(1); recall=tp/cm.sum(1).clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-12); support=cm.sum(1).float()
    return {"accuracy":float(tp.sum()/cm.sum().clamp_min(1)),"precision_macro":float(precision.mean()),"recall_macro":float(recall.mean()),"f1_macro":float(f1.mean()),"f1_weighted":float((f1*support).sum()/support.sum().clamp_min(1)),"confusion_matrix":cm.tolist()}
