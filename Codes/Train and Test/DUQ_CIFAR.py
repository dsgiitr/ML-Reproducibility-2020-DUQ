import argparse
import json
import pathlib
import random

import torch
import torch.nn.functional as F
import torch.utils.data

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from utils.resnet_duq import ResNet_DUQ
from utils.datasets import all_datasets
from utils.evaluate_ood import get_cifar_svhn_ood, get_auroc_classification


model={}
results=[]
def main(
    batch_size,
    epochs,
    length_scale,
    centroid_size,
    model_output_size,
    learning_rate,
    l_gradient_penalty,
    gamma,
    weight_decay,
    final_model, 
    input_dep_ls,
    use_grad_norm
):
    
    # Dataset prep
    ds = all_datasets["CIFAR10"]()
    input_size, num_classes, dataset, test_dataset = ds

    # Split up training set
    idx = list(range(len(dataset)))
    random.shuffle(idx)

    if final_model:
        train_dataset = dataset
        val_dataset = test_dataset
    else:
        val_size = int(len(dataset) * 0.8)
        train_dataset = torch.utils.data.Subset(dataset, idx[:val_size])
        val_dataset = torch.utils.data.Subset(dataset, idx[val_size:])

        val_dataset.transform = (
            test_dataset.transform
        ) 
    kwargs = {"num_workers": 4, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    
    # Model
    global model
    model = ResNet_DUQ(
        input_size, num_classes, centroid_size, model_output_size, length_scale, gamma
    )
    
    model = model.cuda()
    #model.load_state_dict(torch.load("DUQ_CIFAR_75.pt"))

    # Optimiser
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50, 75], gamma=0.2
    )

    def bce_loss_fn(y_pred, y):
        bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
            num_classes * y_pred.shape[0]
        )
        return bce

    def output_transform_bce(output):
        y_pred, y, x = output

        y = F.one_hot(y, num_classes).float()

        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, x = output

        return y_pred, y

    def output_transform_gp(output):
        y_pred, y, x = output

        return x, y_pred

    def calc_gradients_input(x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

    def calc_gradient_penalty(x, y_pred):
        gradients = calc_gradients_input(x, y_pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def step(engine, batch):
        model.train()

        optimizer.zero_grad()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        if l_gradient_penalty > 0:
            x.requires_grad_(True)

        z, y_pred = model(x)
        y = F.one_hot(y, num_classes).float()

        loss = bce_loss_fn(y_pred, y)

        # Avoid calc of computing
        if l_gradient_penalty > 0:
            loss += l_gradient_penalty * calc_gradient_penalty(x, y_pred)

        if use_grad_norm:
            #gradient normalization
            loss/=(1+l_gradient_penalty)

        loss.backward()
        optimizer.step()

        x.requires_grad_(False)

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()


    trainer = Engine(step)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):

        # logging every 10 epoch or last epochs
        if trainer.state.epoch % 10 == 0 or trainer.state.epoch > epochs-5:
          
            #acc on cifar test set and auroc on cifar+svhn testsets
            testacc,auroc_cifsv = get_cifar_svhn_ood(model)
          
            #acc on cifar val set and self auroc on cifar valset
            val_acc, self_auroc = get_auroc_classification(val_dataset, model)

            print(f"Test Accuracy: {testacc}, AUROC: {auroc_cifsv}")
            print(f"AUROC - uncertainty: {self_auroc}, Val Accuracy : {val_acc}")

        scheduler.step()

        # save
        if trainer.state.epoch == epochs-1:
            torch.save(
                model.state_dict(), f"model_{trainer.state.epoch}.pt"
            )

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)
    trainer.run(train_loader, max_epochs=epochs)
    
    testacc,auroc_cifsv = get_cifar_svhn_ood(model)
    val_acc, self_auroc = get_auroc_classification(val_dataset, model)

    return testacc,auroc_cifsv,val_acc, self_auroc
    
if __name__ == "__main__":

    final_model = True  # Train on full train dataset 
    input_dep_ls = False #input dependent length scale (sigma)
    use_grad_norm = False #gradient normalization
    
    repitition = 1
    test_acc_l=[]
    auroc_cifsv_l=[]
    val_acc_l=[]
    self_auroc_l=[]
    
    for i in range(repitition):
        print(f"run {i}\n")

        testacc,auroc_cifsv,val_acc, self_auroc= main(batch_size=128, 
              epochs=75,
              length_scale=0.1,
              centroid_size=512,
              model_output_size=512,
              learning_rate=0.05,
              l_gradient_penalty=0,         
              gamma=0.999,
              weight_decay=5e-4,
              final_model=final_model,
              input_dep_ls = input_dep_ls, 
              use_grad_norm=use_grad_norm)
        test_acc_l.append(testacc)
        auroc_cifsv_l.append(auroc_cifsv)
        val_acc_l.append(val_acc)
        self_auroc_l.append(self_auroc)
    
    print([
                ("val acc", np.mean(val_acc_l),np.std(val_acc_l)),
                ("test acc", np.mean(test_acc_l), np.std(test_acc_l)),
                ("CIFAR_SVHN auroc", np.mean(auroc_cifsv_l), np.std(auroc_cifsv_l)),
                ("Self auroc", np.mean(self_auroc_l), np.std(self_auroc_l)),
            ])
    
    torch.save(model.state_dict(), "DUQ_CIFAR_75.pt")
    print(results)
    