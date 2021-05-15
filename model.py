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


def embedding_module(input_channels=3):
    return nn.Sequential(
        EmbeddingBlock(input_channels),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        nn.Flatten(start_dim=1)
    )


class PrototypicalNetwork(nn.Module):

    def __init__(self, num_classes: int = None, get_probabilities: bool = False, input_channels=3):
        super().__init__()
        self.get_probabilities = get_probabilities
        self.num_classes = num_classes
        self.embedding_nn = embedding_module(input_channels)

    def forward(self, X_supp, y_supp, X_query):
        num_classes = self.num_classes or (y_supp.max() + 1)
        bs, supp_size, c, h, w = X_supp.shape
        tg.guard(X_supp, "*, SUPP_SIZE, C, H, W")
        tg.guard(y_supp, "*, SUPP_SIZE")
        tg.guard(X_query, "*, QUERY_SIZE, C, H, W")
        query_size = X_query.shape[1]
        X_supp = X_supp.flatten(0, 1)
        X_query = X_query.flatten(0, 1)
        embeddings_supp = self.embedding_nn(X_supp).view(bs, supp_size, -1)
        embeddings_query = self.embedding_nn(X_query).view(bs, query_size, -1)
        tg.guard(embeddings_supp, "*, SUPP_SIZE, NUM_FEATURES")
        tg.guard(embeddings_query, "*, QUERY_SIZE, NUM_FEATURES")
        centroids = torch.zeros(bs, num_classes, embeddings_supp.shape[-1], device=X_supp.device)\
                    .scatter_add(1, y_supp, embeddings_supp) / supp_size * num_classes
        tg.guard(centroids, "*, NUM_CLASSES, NUM_FEATURES")
        result = dict(centroids=centroids,
                      embeddings_support=embeddings_supp,
                      embeddings_query=embeddings_query)
        if self.get_probabilities:
            result['prob_query'] = self._get_probabilities(result)
        return result

    def predict_proba(self, X_supp, y_supp, X_query):
        """
        Return probabilities
        :param X_supp:
        :param y_supp:
        :param X_query:
        :return:
        """
        pred = self(X_supp, y_supp, X_query)
        prob = self._get_probabilities(pred)
        return prob

    def _get_probabilities(self, pred):
        """
        
        :param pred: Prediction dictionary of forward method. Must include keys 'embeddings_query' and 'centroids'
        :return: probabilities - Shape: [batch_size, num_classes, query_size]
        :rtype: torch.Tensor
        """
        distance_matrix = (pred['embeddings_query'].unsqueeze(1) -
                           pred['centroids'].unsqueeze(2)) \
            .pow(2)  # [batch_size, num_classes, query_size, emb_features]
        log_prob_unscaled = (- distance_matrix).sum(dim=2)
        const_norm = log_prob_unscaled.logsumexp(dim=1, keepdim=True)
        log_prob = log_prob_unscaled - const_norm
        prob = log_prob.exp()
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
        :rtype: torch.Tensor of shape [num_observarions, query_size]
        """
        return self.predict_proba(X_supp, y_supp, X_query).argmax(dim=1)