# Prototypical Networks
Pytorch implementation of [Prototypical Networks Paper](https://arxiv.org/abs/1703.05175)
## Reproduce Environment

To reproduce the working environment for this project 
create a new conda environment with the following command:

    conda env create -f environment.yaml  #environment with cuda dependencies
    # to create an environment without cuda: conda env create -f environment-cpuonly.yaml

If you don't have the command _conda_ install it  from
[anaconda](https://www.anaconda.com/) or [miniconda](https://conda.io/miniconda.html)

## Repo results

| Dataset      |   Train num classes |   Train support samples |   Train query samples |   Test num classes |   Test support samples |   Test query samples | Paper Accuracy   | Test accuracy   |   Test loss |
|:-------------|--------------------:|------------------------:|----------------------:|-------------------:|-----------------------:|---------------------:|:-----------------|:----------------|------------:|
| cub          |                  50 |                       0 |                    10 |                 50 |                      0 |                   10 | 54.60%           | 53.87%          |   13.5089   |
| miniimagenet |                  20 |                       5 |                    15 |                  5 |                      1 |                    1 | 49.42%           | 45.35%          |   28.6325   |
| miniimagenet |                  20 |                       5 |                    15 |                  5 |                      5 |                    5 | 68.20%           | 66.60%          |   13.6065   |
| miniimagenet |                  30 |                       1 |                    15 |                  5 |                      1 |                    1 | 49.42%           | 49.75%          |   20.1425   |
| miniimagenet |                  30 |                       1 |                    15 |                  5 |                      5 |                    5 | 68.20%           | 63.91%          |   15.5581   |
| omniglot     |                  60 |                       1 |                     5 |                  5 |                      1 |                    1 | 98.80%           | 98.73%          |    0.809822 |
| omniglot     |                  60 |                       1 |                     5 |                  5 |                      5 |                    5 | 99.70%           | 99.67%          |    0.219201 |
| omniglot     |                  60 |                       1 |                     5 |                 20 |                      1 |                    1 | 96.00%           | 95.74%          |    2.65106  |
| omniglot     |                  60 |                       1 |                     5 |                 20 |                      5 |                    5 | 98.90%           | 99.01%          |    0.61366  |
| omniglot     |                  60 |                       5 |                     5 |                  5 |                      1 |                    1 | 98.80%           | 97.79%          |    1.76968  |
| omniglot     |                  60 |                       5 |                     5 |                  5 |                      5 |                    5 | 99.70%           | 99.65%          |    0.282353 |
| omniglot     |                  60 |                       5 |                     5 |                 20 |                      1 |                    1 | 96.00%           | 93.55%          |    5.15215  |
| omniglot     |                  60 |                       5 |                     5 |                 20 |                      5 |                    5 | 98.90%           | 98.84%          |    0.794082 |


- Download [link](https://mega.nz/file/GKI1DQxb#BNIlgnbwlmwJm7IYF3gk-4agVAZZOEdauQ5PrjEL_1Y) of table checkpoints


### CUB Dataset

To reproduce the results of Zero Shot CUB dataset download the preprocessed dataset
at this [link](https://mega.nz/file/iDpXCCaL#j5AI-LKKJqgygjIsWtBN1Ow_-yDs1f36Ki8PHtesgB0).

The archive contains GoogleLeNet visual features of CUB images and class features.


### Resume training

To resume an interrupted training use the argument _--run-path_ of _train.py_. 
An example is: `python train.py --run-path=run/0`

### Project structure

Project structure is very straightforward and flatten to maintain simplicity.

    ├── data  # where data will be downloaded
    │   └── omniglot   #contains omniglot data
    ├── environment-cpuonly.yaml  #conda cpu environmnet
    ├── environment.yaml   # conda environmnet to run this project
    ├── logger.py   #  logging function with tensorboard
    ├── Makefile   
    ├── model.py
    ├── paper.pdf
    ├── paths.py
    ├── readme.md
    ├── run  # where experiments will be stored 
    │   └── 0   # experiments are stored using index based folder name
    ├── test.py
    ├── trainer.py
    ├── train.py
    └── utils.py


### Custom Train 
If you want to train by yourself the model you can train directly by running `python3 train.py`.
The script supports many arguments which are:
    
    usage: train.py [-h] --dataset {omniglot,miniimagenet,cub} --classes NUM_CLASSES --support-samples SUPPORT_SAMPLES [--query-samples QUERY_SAMPLES] [--distance {euclidean,cosine}] [--epochs EPOCHS]
                    [--epoch-steps EPOCH_STEPS] [--seed SEED] [--device DEVICE] [--batch-size BATCH_SIZE] [--eval-steps EVAL_STEPS] [--run-path RUN_PATH]
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset {omniglot,miniimagenet,cub}, -d {omniglot,miniimagenet,cub}
                            Specify train dataset
      --classes NUM_CLASSES, --num-classes NUM_CLASSES, -c NUM_CLASSES
                            Number of classes for each task in meta learning i.e. the N in N-way with K shots
      --support-samples SUPPORT_SAMPLES, -s SUPPORT_SAMPLES
                            Number of training samples for each class in meta learning i.e. the K in N-way with K shots
      --query-samples QUERY_SAMPLES, -q QUERY_SAMPLES
                            Number of test samples for each class in meta learning
      --distance {euclidean,cosine}, --dst {euclidean,cosine}
                            Distance function to use inside PrototypicalNetwork
      --epochs EPOCHS, -e EPOCHS
                            Number of training epochs. Set by default to a very high value because paper specify that train continues until validation loss continues to decrease.
      --epoch-steps EPOCH_STEPS
      --seed SEED
      --device DEVICE
      --batch-size BATCH_SIZE
      --eval-steps EVAL_STEPS
                            Number of evaluation steps. By default is set to the number of steps to reach 600 episodes considering batch size as paper reports
      --run-path RUN_PATH   Set to resume a checkpoint

### Custom Test

You can also test by yourself running `python3 test.py`. It supports also many arguments which are

    usage: test.py [-h] --run-path RUN_PATH [--support-size SUPPORT_SAMPLES] [--query-size QUERY_SAMPLES] [--num-classes NUM_CLASSES] [--batch-size BATCH_SIZE] [--seed SEED] [--steps STEPS] [--device DEVICE]
    
    optional arguments:
      -h, --help            show this help message and exit
      --run-path RUN_PATH, -r RUN_PATH
      --support-size SUPPORT_SAMPLES, -s SUPPORT_SAMPLES
      --query-size QUERY_SAMPLES, -q QUERY_SAMPLES
      --num-classes NUM_CLASSES, --nc NUM_CLASSES
      --batch-size BATCH_SIZE, -b BATCH_SIZE
      --seed SEED
      --steps STEPS
     

