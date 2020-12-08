## [ML-Reproducibility-2020-DUQ](https://paperswithcode.com/rc2020)

# [ [RE] Uncertainty Estimation Using a Single Deep Deterministic Neural Network](https://arxiv.org/abs/2003.02037)

This repository is the reproduction of paper "Uncertainty Estimation Using a Single Deep Deterministic Neural Network" by Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal. 

All codes for training and experiments are provided. They are as colab notebooks. Additionally, py files are provided for training and testing.<br>

## Requirements

### Colab notebooks
All requirements will be self-installed 

### py files

To install requirements:
```setup
pip install -r requirements.txt
```
All datasets will be downloaded from torchvision except for notMNIST <br>
Download it from "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat" and place it in "data" folder:
```
mkdir -p data && cd data && curl -O "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat"
```

## Running Experiments and Training, Testing

### Colab notebooks
* Open a notebook
* Place the utils folder (only requirement)
* Place the trained models, specify the location of trained model in code (for experiments)
* Run the notebook

### py files
* Create a virtual env
* Download the requirements
* Change hyper parameters and Run the python files


Training: will be done on FMnist or CIFAR-10 <br>
Testing: Accuracy will be calculated on testset of FMnist or CIFAR-10 <br>
Testing: Auroc-ood will be calculated on FMnist+Mnist-ood or CIFAR10+SVHN-ood <br>
Trained models are required to execute the experiment codes

## Pre-trained Models

You can download pretrained models here:

- [DUQ and DE ](https://drive.google.com/drive/folders/1WSmDiCDlnQT3oUmeLwsfydQDBGP3E6QY?usp=sharing) trained on FashionMNIST, CIFAR10 with paper's default parameters 


## Results

Performance of the model with our implementation at a glance:

### FashionMNIST, Mnist-ood

| Model name         | Accuracy on FM | Auroc-ood(M) | Train, Test Time*|
| ------------------ |--------------- | ------------ | ---------------- |
| DUQ with gp        |     92.34%     |     0.9411   |   23s and 1s     |
| DE                 |     93.30%     |     0.8890   |   9x5s and 2.3s  |


### CIFAR10, SVHN-ood

| Model name         | Accuracy on CIFAR10 | Auroc-ood(M) | Train, Test Time*|
| ------------------ |-------------------- | ------------ | ---------------- |
| DUQ with gp        |       93.45%        |     0.9313   |   210s and 4s    |
| DE                 |       94.44%        |     0.9489   |   60x5s and 14s  |

## Reference
```
@misc{vanamersfoort2020uncertainty,
      title={Uncertainty Estimation Using a Single Deep Deterministic Neural Network}, 
      author={Joost van Amersfoort and Lewis Smith and Yee Whye Teh and Yarin Gal},
      year={2020},
      eprint={2003.02037},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


