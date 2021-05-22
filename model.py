import torch
from torch import nn
import tensorguard as tg

def EmbeddingBlock(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, ceil_mode=False)
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

    def __init__(self, distance_function, num_classes: int = None, get_probabilities: bool = False, input_channels=3):
        super().__init__()
        self.get_probabilities = get_probabilities
        self.num_classes = num_classes
        self.embedding_nn = embedding_module(input_channels)
        self.distance_function = distance_function

    def forward(self, X_supp, y_supp, X_query):
        num_classes = y_supp.max() + 1
        bs, supp_size, c, h, w = X_supp.shape
        tg.guard(X_supp, "*, SUPP_SIZE, C, H, W")
        tg.guard(y_supp, "*, SUPP_SIZE")
        tg.guard(X_query, "*, QUERY_SIZE, C, H, W")
        query_size = X_query.shape[1]
        X_supp = X_supp.flatten(0, 1).contiguous()
        X_query = X_query.flatten(0, 1).contiguous()
        embeddings_supp = self.embedding_nn(X_supp).view(bs, supp_size, -1)
        embeddings_query = self.embedding_nn(X_query).view(bs, query_size, -1)
        tg.guard(embeddings_supp, "*, SUPP_SIZE, NUM_FEATURES")
        tg.guard(embeddings_query, "*, QUERY_SIZE, NUM_FEATURES")
        y_supp_broadcast = y_supp.unsqueeze(-1).expand(bs, supp_size, embeddings_supp.shape[-1])
        centroids = torch.zeros(bs, num_classes, embeddings_supp.shape[-1], device=X_supp.device, dtype=embeddings_supp.dtype)\
                    .scatter_add(1, y_supp_broadcast, embeddings_supp) / supp_size * num_classes
#         tg.guard(centroids, "*, NUM_CLASSES, NUM_FEATURES")
        result = dict(centroids=centroids,
                      embeddings_support=embeddings_supp,
                      embeddings_query=embeddings_query)
        if self.get_probabilities:
            result['prob_query'] = self._get_probabilities(result, self.distance_function)
        return result

    def predict_proba(self, X_supp, y_supp, X_query):
        """
        Return probabilities
        :param X_supp:
        :param y_supp:
        :param X_query: 
        :return: Tensor of probabilities - Shape: [batch_size, num_classes, query_size]
        """
        pred = self(X_supp, y_supp, X_query)
        prob = self._get_probabilities(pred, self.distance_function)
        return prob

    @staticmethod
    def _get_probabilities(pred, distance_function):
        """
        
        :param pred: Prediction dictionary of forward method. Must include keys 'embeddings_query' and 'centroids'
        :return: probabilities - Shape: [batch_size, num_classes, query_size]
        :rtype: torch.Tensor
        """
        distance_matrix = distance_function(pred['embeddings_query'].unsqueeze(1), pred['centroids'].unsqueeze(2)) # [batch_size, num_classes, query_size, emb_features]
#         tg.guard(distance_matrix, "*, NUM_CLASSES, QUERY_SIZE, NUM_FEATURES")
        log_prob_unscaled = (- distance_matrix).sum(dim=-1)
        const_norm = log_prob_unscaled.logsumexp(dim=1, keepdim=True)
        log_prob = log_prob_unscaled - const_norm
        prob = log_prob.exp()
#         tg.guard(prob, "*, NUM_CLASSES, QUERY_SIZE")
        return prob

    def predict(self, X_supp, y_supp, X_query):
        """
        :param X_supp: train samples in a meta-learning task
        :type X_supp: torch.Tensor
        :param y_supp: train labels in a meta-learning task
        :type y_supp: torch.Tensor
        :param X_query: test samples in a meta-learning task
        :type X_query: torch.Tensor
        :return: The predicted class for each X_query samples - Shape [num_observarions, query_size]
        :rtype: torch.Tensor
        """
        return self.predict_proba(X_supp, y_supp, X_query).argmax(dim=1)


class PrototypicalNetworkZeroShot(nn.Module):

    def __init__(self, distance_function, num_classes: int = None, get_probabilities: bool = False,
                 meta_features: int = 312, img_features: int = 1024, eps: float = 1e-6):
        super().__init__()
        self.distance_function = distance_function
        self.meta_features = meta_features
        self.img_features = img_features
        self.get_probabilities = get_probabilities
        self.num_classes = num_classes
        self.linear_img = nn.Linear(img_features, img_features)
        self.linear_meta = nn.Linear(meta_features, img_features)
        self.eps = eps

    def forward(self, meta_classes, X_query):
        centroids = self.linear_meta(meta_classes)
        centroids = centroids / (centroids.norm(2, dim=-1, keepdim=True) + self.eps)
        embeddings_query = self.linear_img(X_query)
        result = dict(centroids=centroids,
                    embeddings_query=embeddings_query)
        if self.get_probabilities:
            result['prob_query'] = PrototypicalNetwork._get_probabilities(result, self.distance_function)
        return result

    def predict_proba(self, meta_classes, X_query):
        pred = self(meta_classes, X_query)
        return PrototypicalNetwork._get_probabilities(pred, self.distance_function)

    def predict(self, meta_classes, X_query):
        """
        :param meta_classes: metadata features for classes - Shape [batch_size, num_classes, num_metadata_features]
        :type X_supp: torch.Tensor
        :param X_query: test samples in a meta-learning task
        :type X_query: torch.Tensor
        :return: The predicted class for each X_query samples - Shape [num_observarions, query_size]
        :rtype: torch.Tensor
        """
        return self.predict_proba(meta_classes, X_query).argmax(dim=1)
