import torch
from torch.nn import Conv2d, ReLU, Sequential, MaxPool2d, Flatten, Module, BatchNorm2d


def EmbeddingBlock(input_channels):
    return Sequential(
        Conv2d(input_channels, 64, kernel_size=3, padding=1),
        BatchNorm2d(64),
        ReLU(),
        MaxPool2d(2, ceil_mode=False)
    )


def embedding_miniimagenet():
    return Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        Flatten(start_dim=1)
    )


def embedding_omniglot():
    return Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        Flatten(start_dim=1)
    )


class PrototypicalNetwork(Module):

    def __init__(self, distance_f='euclidean'):
        super().__init__()
        self.embedding_nn = embedding_miniimagenet()
        if distance_f == 'euclidean':
            self.distance_f = lambda x, y: (x - y).norm(2, dim=-1)

    def forward(self, batch_supp, y_train, batch_query):
        batch_size, supp_size = batch_supp.shape[:2]
        query_size = batch_query.size(1)
        num_classes = y_train.max() + 1
        batch_supp = batch_supp.reshape(batch_supp.size(0) *  # bs
                                        batch_supp.size(1),  # n_s
                                        batch_supp.size(2),  # channel
                                        batch_supp.size(3),  # w
                                        batch_supp.size(4))  # h
        embeddings_supp = self.embedding_nn(batch_supp)
        embeddings_supp = embeddings_supp.reshape(batch_size, supp_size, num_classes, -1)
        c = torch.zeros(batch_size, num_classes, embeddings_supp.shape[-1]).to(batch_supp.device)
        for i_batch in range(batch_size):
            for i_supp in range(supp_size):
                    c[i_batch, y_train[i_batch, i_supp]] += embeddings_supp[i_batch, i_supp]
        c /= supp_size
        batch_query = batch_query.reshape(batch_query.size(0) *
                                          batch_query.size(1),
                                          batch_query.size(2),
                                          batch_query.size(3),
                                          batch_query.size(4))
        embeddings_query = self.embedding_nn(batch_query)
        embeddings_query = embeddings_query.reshape(batch_size, query_size, num_classes, -1)
        return - self.distances_centers(c, embeddings_query).log_softmax(dim=-1)

    def distances_centers(self, c, query):
        c = c.unsqueeze(1)
        query = query.unsqueeze(2)
        return self.distance_f(query, c)


