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

- Run 600 episodes


| Dataset       | Train num classes | Train support samples | Train query samples | Test num classes | Test support samples | Test query samples | Test accuracy | Test loss |
|---------------|-------------------|-----------------------|---------------------|------------------|----------------------|--------------------|---------------|-----------|
| Mini Imagenet | 20                | 5                     | 15                  | 5                | 1                    | 1                  | 41.36%        | 41.76     |
| Mini Imagenet | 20                | 5                     | 15                  | 5                | 5                    | 5                  | 64.92%        | 14.64     |
| Mini Imagenet | 30                | 1                     | 15                  | 5                | 1                    | 1                  | 45.02%        | 21.26     |
| Mini Imagenet | 30                | 1                     | 15                  | 5                | 5                    | 5                  | 60.55%        | 17.75     |
| Omniglot      | 60                | 1                     | 5                   | 5                | 1                    | 1                  | 98.73%        | 0.81      |
| Omniglot      | 60                | 1                     | 5                   | 5                | 5                    | 5                  | 99.67%        | 0.22      |
| Omniglot      | 60                | 1                     | 5                   | 20               | 1                    | 1                  | 95.74%        | 2.65      |
| Omniglot      | 60                | 1                     | 5                   | 20               | 5                    | 5                  | 99.01%        | 0.61      |
| Omniglot      | 60                | 5                     | 5                   | 5                | 1                    | 1                  | 97.79%        | 1.77      |
| Omniglot      | 60                | 5                     | 5                   | 5                | 5                    | 5                  | 99.65%        | 0.28      |
| Omniglot      | 60                | 5                     | 5                   | 20               | 1                    | 1                  | 93.55%        | 5.15      |
| Omniglot      | 60                | 5                     | 5                   | 20               | 5                    | 5                  | 98.84%        | 0.79      |

## CUB Dataset

To reproduce the results of Zero Shot CUB dataset download the preprocessed dataset
at this [link](https://mega.nz/file/iDpXCCaL#j5AI-LKKJqgygjIsWtBN1Ow_-yDs1f36Ki8PHtesgB0).

It contains GoogleLeNet visual features of CUB images and class features 
used as metadata features.


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
     

