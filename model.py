from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential, MaxPool2d


def EmbeddingBlock(input_channels):
    return Sequential(
        Conv2d(input_channels, 64, kernel_size=3),
        BatchNorm2d(64),
        ReLU(),
        MaxPool2d(2)
    )

def embedding_miniimagenet():
    return Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64)
    )


def embedding_omniglot():
    return EmbeddingBlock(3)