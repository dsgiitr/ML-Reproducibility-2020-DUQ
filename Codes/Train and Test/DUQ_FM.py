import random
import numpy as np

import torch
import torch.utils.data
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.evaluate_ood import (
    get_fashionmnist_mnist_ood,
    get_fashionmnist_notmnist_ood,
)
from utils.datasets import FastFashionMNIST, get_FashionMNIST
from utils.cnn_duq import CNN_DUQ

model={}
def train_model(l_gradient_penalty, length_scale, final_model,epochs):

    input_size = 28
    num_classes = 10
    embedding_size = 256
    learnable_length_scale = False #Learnable length scale
    gamma = 0.999
    input_dep_ls = False #input dependent length scale (sigma)

    if input_dep_ls and learnable_length_scale: #only one can be True
        learnable_length_scale=False


    ## Main (FashionMNIST) and ood (Mnist) Dataset
    dataset = FastFashionMNIST("data/", train=True, download=True)
    test_dataset = FastFashionMNIST("data/", train=False, download=True)

    idx = list(range(60000))
    random.shuffle(idx)

    if final_model:
        train_dataset = dataset
        val_dataset = test_dataset
    else:
        train_dataset = torch.utils.data.Subset(dataset, indices=idx[:55000])
        val_dataset = torch.utils.data.Subset(dataset, indices=idx[55000:])

    dl_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
    )

    dl_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=2000, shuffle=False, num_workers=0
    )

    dl_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=2000, shuffle=False, num_workers=0
    )


    # Model
    global model
    model = CNN_DUQ(
        input_size,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
        input_dep_ls 
    )
    
    model = model.cuda()
    #model.load_state_dict(torch.load("DUQ_FM_30_FULL.pt"))

    # Optimiser
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4
    )

    def output_transform_bce(output):
        y_pred, y, _, _ = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, _, _ = output
        return y_pred, torch.argmax(y, dim=1)

    def output_transform_gp(output):
        y_pred, y, x, y_pred_sum = output
        return x, y_pred_sum

    def calc_gradient_penalty(x, y_pred_sum):
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        z, y_pred = model(x)

        loss = F.binary_cross_entropy(y_pred, y)
        loss += l_gradient_penalty * calc_gradient_penalty(x, y_pred.sum(1))
        
        #gradient normalization
        #loss/=(1+l_gradient_penalty) 
        x.requires_grad_(False)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        z, y_pred = model(x)

        return y_pred, y, x, y_pred.sum(1)

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gradient_penalty")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.2
    )

    pbar = ProgressBar()
    pbar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        scheduler.step()

        # logging every 5 epoch
        if trainer.state.epoch % 5 == 0:
            evaluator.run(dl_val)

            # AUROC on FashionMNIST + Mnist / NotMnist
            accuracy, roc_auc_mnist = get_fashionmnist_mnist_ood(model)
            accuracy, roc_auc_notmnist = get_fashionmnist_notmnist_ood(model)
            metrics = evaluator.state.metrics

            print(
                f"Validation Results - Epoch: {trainer.state.epoch} "
                f"Val_Acc: {metrics['accuracy']:.4f} "
                f"BCE: {metrics['bce']:.2f} "
                f"GP: {metrics['gradient_penalty']:.6f} "
                f"AUROC MNIST: {roc_auc_mnist:.4f} "
                f"AUROC NotMNIST: {roc_auc_notmnist:.2f} "
            )
            print(f"Sigma: {model.sigma}")

    # Train
    trainer.run(dl_train, max_epochs=epochs)

    # Validation
    evaluator.run(dl_val)
    val_accuracy = evaluator.state.metrics["accuracy"]

    # Test
    evaluator.run(dl_test)
    test_accuracy = evaluator.state.metrics["accuracy"]

    return model, val_accuracy, test_accuracy

if __name__ == "__main__":
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()

    l_gradient_penalties = [0.05]
    length_scales = [0.1]
    epochs=30

    repetition = 1  # Increase for multiple repetitions
    final_model = True  # set true for final model to train on full train set

    results = {}

    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:
            val_accuracies = []
            test_accuracies = []
            roc_aucs_mnist = []
            roc_aucs_notmnist = []

            for _ in range(repetition):
                print(" ### NEW MODEL ### ")
                model, val_accuracy, test_accuracy = train_model(
                    l_gradient_penalty, length_scale, final_model, epochs
                )
                accuracy, roc_auc_mnist = get_fashionmnist_mnist_ood(model)
                _, roc_auc_notmnist = get_fashionmnist_notmnist_ood(model)

                val_accuracies.append(val_accuracy)
                test_accuracies.append(test_accuracy)
                roc_aucs_mnist.append(roc_auc_mnist)
                roc_aucs_notmnist.append(roc_auc_notmnist)
            
            # All stats
            results[f"lgp{l_gradient_penalty}_ls{length_scale}"] = [
                ("val acc", np.mean(val_accuracies)),
                ("test acc", np.mean(test_accuracies)),
                ("M auroc", np.mean(roc_aucs_mnist)),
                ("NM auroc", np.mean(roc_aucs_notmnist)),
            ]
            #print(results[f"lgp{l_gradient_penalty}_ls{length_scale}"])
    
    # Save
    torch.save(model.state_dict(), "DUQ_FM_30_FULL.pt")
    print(results)
