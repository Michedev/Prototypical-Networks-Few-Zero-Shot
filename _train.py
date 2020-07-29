from ignite.engine import Engine
import torch


def calc_loss(model, n: int, n_s: int, batch):
    batch_size = batch.size(0)
    batch_supp = batch[: :n_s]
    batch_query = batch[: n_s:]
    batch_supp = batch_supp.view(batch_supp.size(0) * 
                                 batch_supp.size(1) * 
                                 batch_supp.size(2), 
                                 batch_supp.size(3), 
                                 batch_supp.size(4),
                                 batch_supp.size(5))
    embeddings_supp = model(batch_supp)
    embeddings_supp = embeddings_supp.view(batch_size, n_s, n, -1)
    c: torch.Tensor = embeddings_supp.mean(dim=1).detach()
    loss = 0.0
    batch_query = batch_query.view(batch_query.size(0) * 
                                    batch_query.size(1) * 
                                    batch_query.size(2), 
                                    batch_query.size(3), 
                                    batch_query.size(4),
                                    batch_query.size(5))
    embeddings_query = model(batch_query)
    embeddings_query = embeddings_query.view(batch_size, n - n_s, n, -1)
    loss = (embeddings_query - c).pow(2).mean(dim=0).sum().sqrt()
    for i in range(batch_size):
        for j in range(n - n_s):
            for k in range(n):
                loss += embeddings_query[i, j, k] - c

    
    

def train(model, opt, epochs: int, n_s: int, train_it, test_it, half_lr: bool = True):
    pass
    