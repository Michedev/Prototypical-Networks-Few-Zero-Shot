# Prototypical Networks
Pytorch implementation of [Prototypical Networks Paper](https://arxiv.org/abs/1703.05175)
## Mini-Imagenet results

- Loaded from _model_weights/embedding.pth_
- Run 600 episodes

|   |Mean|Std|
|---|---|---|
|_Test loss_|4.01|1.31|
|_Test accuracy_|0.503|0.101|

## How to run
_Makefile_ contains the same parameters that I've used to train and get test results. <br>
- To train the model for mini-imagenet just digit into a terminal `make train-miniimagenet`
- To get test results with trained model at path _model_weights/embedding.pth_ digit `make test-miniimagenet`
### Custom Train 
If you want to train by yourself the model you can train directly by running `python3 train.py`.
The script supports many arguments which are:
    
    SYNOPSIS
        train.py DATASET TRAIN_N VAL_N TEST_N N_S N_Q <flags>
    
    DESCRIPTION
        Train the model and save under model_weights both last epoch moodel (model_weights/embedding.pth) and the one with lowest validation loss (model_weights/best_embedding.pth)
    
    POSITIONAL ARGUMENTS
        DATASET
            train dataset; can be 'omniglot' or 'miniimagenet' [str]
        TRAIN_N
            num classes in train split (i.e. n in meta learning) [int]
        VAL_N
            num classes in val split (i.e. n in meta learning) [int]
        TEST_N
            num classes in test split (i.e. n in meta learning) [int]
        N_S
            size of support set for each task (see paper for more details) [int]
        N_Q
            size of query set for each task (see paper for more details) [int]
    
    FLAGS
        --epochs=EPOCHS
            Num epochs of training [int]
        --batch_size=BATCH_SIZE
            Batch size [int]
        --lr=LR
            learning rate [float]
        --trainsize=TRAINSIZE
            Size of training set. Remember thought that instances are sampled randomly therefore it's useful only to set switch between training and validation. [int]
        --valsize=VALSIZE
            Size of validation set. [int]
        --testsize=TESTSIZE
            Size of test set. [int]
        --device=DEVICE
            location of data and model parameters. Can be 'cpu' or 'cuda:*'
    
    NOTES
        You can also use flags syntax for POSITIONAL ARGUMENTS

### Custom Test

You can also test by yourself running `python3 test.py`. It supports also many arguments which are

    NAME
        test.py - Evaluate model performance into test split and get printed loss and accuracy

    SYNOPSIS
        test.py DATASET N N_S N_Q TESTSIZE <flags>
    
    DESCRIPTION
        Evaluate model performance into test split and get printed loss and accuracy
    
    POSITIONAL ARGUMENTS
        DATASET
            train dataset; can be 'omniglot' or 'miniimagenet' [str] (Note: actually works only with 'miniimagenet')
        N
            num classes for each meta task [int]
        N_S
            size of support set for each task (see paper for more details) [int]
        N_Q
            size of query set for each task (see paper for more details) [int]
        TESTSIZE
            num of episodes of test split
    
    FLAGS
        --use_best=USE_BEST
            instead of loading last epoch model parameters load the one with the lowest validation loss [bool] [default False]
        --device=DEVICE
            Location of data and model parameters storage [string] [default 'cpu']
    
    NOTES
        You can also use flags syntax for POSITIONAL ARGUMENTS
