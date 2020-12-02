import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.datasets import all_datasets
from utils.cnn_duq import SoftmaxModel as CNN
from torchvision.models import resnet18
from utils.resnet import ResNet
from utils.ensemble_eval import (get_fm_mnist_ood_ensemble, get_cifar10_svhn_ood_ensemble)


def train(model, train_loader, optimizer, epoch, loss_fn):

    ##Train function##########

    model.train()

    total_loss = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = loss_fn(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = torch.tensor(total_loss).mean()
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(models, test_loader, loss_fn):    
    
    ##Test function##########

    models.eval()
    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            losses = torch.empty(len(models), data.shape[0])
            predictions = []
            for i, model in enumerate(models):
                predictions.append(model(data))
                losses[i, :] = loss_fn(predictions[i], target, reduction="sum")

            predictions = torch.stack(predictions)

            loss += torch.mean(losses)
            avg_prediction = predictions.exp().mean(0)

            # get the index of the max log-probability
            class_prediction = avg_prediction.max(1)[1]
            correct += (
                class_prediction.eq(target.view_as(class_prediction)).sum().item()
            )

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct


def main():
    
    ## Epochs, lr, Dataset={"FashionMNIST","CIFAR10"}

    args={'epochs':30,'lr':0.05,'ensemble':5,'dataset':"FashionMNIST"}                    
    loss_fn = F.nll_loss


    #Selecting Main Dataset
    #FashionMNIST-Mnist
    #CIFAR10-SVHN
    ds = all_datasets[args['dataset']]()
    input_size, num_classes, train_dataset, test_dataset = ds
    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5000, shuffle=False, **kwargs
    )


    #Selecting model CNN for FashionMNIST and Resnet for CIFAR10

    if args['dataset'] == "FashionMNIST":
        milestones = [10, 20]
        ensemble = [CNN(input_size, num_classes).cuda() for _ in range(args['ensemble'])]
    else:
        milestones = [25, 50]
        ensemble = [
            ResNet(input_size, num_classes).cuda() for _ in range(args['ensemble'])
        ]

    ensemble = torch.nn.ModuleList(ensemble)
    #ensemble.load_state_dict(torch.load("FM_5_ensemble_30.pt"))
    
    optimizers = []
    schedulers = []

    for model in ensemble:
        # Need different optimisers to apply weight decay and momentum properly
        # when only optimising one element of the ensemble
        optimizers.append(
            torch.optim.SGD(
                model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4
            )
        )

        schedulers.append(
            torch.optim.lr_scheduler.MultiStepLR(
                optimizers[-1], milestones=milestones, gamma=0.1
            )
        )

    for epoch in range(1, args['epochs'] + 1):
        #####Train#####
        for i, model in enumerate(ensemble):                                             
            train(model, train_loader, optimizers[i], epoch, loss_fn)
            schedulers[i].step()

        #####Test######
        #Test on testset of main dataset
        test(ensemble, test_loader, loss_fn)   

        #####AUROC######   
        #AUROC on Main + ood                                  
        if(args['dataset'] == "FashionMNIST"):
            accuracy, auroc = get_fm_mnist_ood_ensemble(ensemble)                        
            print({'mnist_ood_auroc':auroc})
        else:
            accuracy, auroc = get_cifar10_svhn_ood_ensemble(ensemble)
            print({'cifar10_ood_auroc':auroc})   

    #Save
    path = f"model{args['dataset']}_{len(ensemble)}"
    torch.save(ensemble.state_dict(), path + "_ensemble.pt")


if __name__ == "__main__":
    main()
