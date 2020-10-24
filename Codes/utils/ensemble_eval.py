import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import (roc_curve, roc_auc_score)

from utils.datasets import (
    get_CIFAR10,
    get_SVHN,
    get_FashionMNIST,
    get_MNIST,
    get_notMNIST,
)
from utils.evaluate_ood import prepare_ood_datasets

def get_fm_mnist_ood_ensemble(ensemble):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()
    
    dataloader, anomaly_targets = prepare_ood_datasets(fashionmnist_test_dataset, mnist_test_dataset)
    scores = []
    accuracies = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()
            output_l=[]
            for i,model in enumerate(ensemble):
                model.eval()
                output = model(data)
                output_l.append(output)
            output_l = torch.stack(output_l)

            kernel_distance=output_l.exp().mean(0)
            pred=kernel_distance.argmax(1)
            kernel_distance=-(kernel_distance*torch.log(kernel_distance)).sum(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())
            scores.append(kernel_distance.cpu().numpy())
    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)
    accuracy = np.mean(accuracies[: len(fashionmnist_test_dataset)])
    auroc = roc_auc_score(anomaly_targets, scores)
    return accuracy, auroc

def get_cifar10_svhn_ood_ensemble(ensemble):
    _, _, _, cifar_test_dataset = get_CIFAR10()
    _, _, _, svhn_test_dataset = get_SVHN()
    
    dataloader, anomaly_targets = prepare_ood_datasets(cifar_test_dataset, svhn_test_dataset)
    scores = []
    accuracies = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()
            output_l=[]
            for i,model in enumerate(ensemble):
                model.eval()
                output = model(data)
                output_l.append(output)
            output_l = torch.stack(output_l)
            kernel_distance=output_l.exp().mean(0)
            pred=kernel_distance.argmax(1)

            kernel_distance=-(kernel_distance*torch.log(kernel_distance)).sum(1)
            
            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())
            scores.append(kernel_distance.cpu().numpy())
    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)
    accuracy = np.mean(accuracies[: len(cifar_test_dataset)])
    auroc = roc_auc_score(anomaly_targets, scores)
    return accuracy, auroc

def get_ROC_mnist_ensemble(ensemble):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()
    
    dataloader, anomaly_targets = prepare_ood_datasets(fashionmnist_test_dataset, mnist_test_dataset)
    scores = []
    accuracies = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()
            output_l=[]
            for i,model in enumerate(ensemble):
                model.eval()
                output = model(data)
                output_l.append(output)
            output_l = torch.stack(output_l)

            kernel_distance=output_l.exp().mean(0)
            pred=kernel_distance.argmax(1)
            
            kernel_distance=-(kernel_distance*torch.log(kernel_distance)).sum(1)
            
            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())
            scores.append(kernel_distance.cpu().numpy())
    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)
    accuracy = np.mean(accuracies[: len(fashionmnist_test_dataset)])
    roc = roc_curve(anomaly_targets, scores)
    return roc