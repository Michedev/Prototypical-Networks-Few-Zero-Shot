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
Makefile contains the same parameters that I've used to train and get test results. <br>
- To train the model for mini-imagenet just digit into a terminal `make train-miniimagenet`
- To get test results with trained model at path _model_weights/embedding.pth_ digit `make test-miniimagenet`
### Deep customization
If you want 