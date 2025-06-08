# coding=utf-8
import torch
import numpy as np
from network import act_network
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def get_fea(args):
    net = act_network.ActNetwork(args.dataset)
    return net


def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            if usedpredict == 'p':
                p = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

def accuracy_metrics(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0
    all_preds = []
    all_labels = []
    all_weights = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.eval()
    
    with torch.no_grad():
        for data in loader:
            x, y = data[0].to(device).float(), data[1].to(device).long()
            
            p = network.predict(x) if usedpredict == 'p' else network.predict1(x)
            
            if weights is None:
                batch_weights = torch.ones(len(x), device=device)
            else:
                batch_weights = weights[weights_offset:weights_offset + len(x)].to(device)
                weights_offset += len(x)
            
            if p.size(1) == 1:
                preds = p.gt(0).squeeze(1)  
            else:
                preds = p.argmax(1)  
            
            correct += (preds.eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_weights.append(batch_weights.cpu())
    
    network.train()
    
    accuracy = correct / total if total > 0 else 0
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_weights = torch.cat(all_weights).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, sample_weight=all_weights
    )
    
    return accuracy, np.mean(precision), np.mean(recall), np.mean(f1)

def confusion_matrix_metrics(network, loader, weights, usedpredict='p', num_classes=None):
    weights_offset = 0
    all_preds = []
    all_labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.eval()
    
    with torch.no_grad():
        for data in loader:
            x, y = data[0].to(device).float(), data[1].to(device).long()
            
            p = network.predict(x) if usedpredict == 'p' else network.predict1(x)
            
            if p.size(1) == 1:
                preds = p.gt(0).squeeze(1)  
            else:
                preds = p.argmax(1)  
            
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    
    network.train()
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    if num_classes is None:
        all_classes = np.unique(np.concatenate((all_preds, all_labels)))
    else:
        all_classes = np.arange(num_classes)  

    cm = confusion_matrix(all_labels, all_preds, labels=all_classes)
    return cm

