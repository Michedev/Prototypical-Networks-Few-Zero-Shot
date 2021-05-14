import torch
from torch import nn
import tensorguard as tg

def EmbeddingBlock(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, ceil_mode=True)
    )


def embedding_module():
    return nn.Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        nn.Flatten(start_dim=1)
    )


class PrototypicalNetwork(nn.Module):

    def __init__(self, num_classes: int = None, calc_p_k=False):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_nn = embedding_module()
        self.calc_p_k = calc_p_k

    def forward(self, X_supp, y_supp, X_query):
        num_classes = self.num_classes or (y_supp.max() + 1)
        bs, supp_size, c, h, w = X_supp.shape
        tg.guard(X_supp, "BS, SUPP_SIZE, C, H, W")
        tg.guard(y_supp, "BS, SUPP_SIZE")
        tg.guard(X_query, "BS, QUERY_SIZE, C, H, W")
        query_size = X_query.shape[1]
        X_supp = X_supp.flatten(0, 1)
        X_query = X_query.flatten(0, 1)
        embeddings_supp = self.embedding_nn(X_supp).view(bs, supp_size, -1)
        embeddings_query = self.embedding_nn(X_query).view(bs, query_size, -1)
        centroids = torch.zeros(bs, num_classes, embeddings_supp.shape[-1], device=X_supp.device)\
                    .scatter_add(1, y_supp, embeddings_supp) / supp_size * num_classes
        result = dict(centroids=centroids,
                      embeddings_support=embeddings_supp,
                      embeddings_query=embeddings_query)
        return result

    def predict_proba(self, X_supp, y_supp, X_query):

        pred = self(X_supp, y_supp, X_query)
        distance_matrix = (pred['embeddings_query'].unsqueeze(1) -
                           pred['centroids'].unsqueeze(2))\
                           .pow(2)  #[batch_size, num_classes, query_size * num_classes, emb_features]
        prob = (- distance_matrix).sum(dim=2).softmax(dim=1)
        return prob

    def predict(self, X_supp, y_supp, X_query):
        """
        :param X_supp: train samples in a meta-learning task
        :type X_supp: torch.Tensor
        :param y_supp: train labels in a meta-learning task
        :type y_supp: torch.Tensor
        :param X_query: test samples in a meta-learning task
        :type X_query: torch.Tensor
        :return: The predicted class for each X_query samples
        :rtype: torch.Tensor of shape [num_observarions,
        """
        return self.predict_proba(X_supp, y_supp, X_query).argmax(dim=1)