from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import torch
import torch.nn as nn
import torchsummary
from umap.umap_ import find_ab_params


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return 1.0 / (1.0 + a * distances ** (2 * b))

def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.log(
        torch.clamp(probabilities_distance, min=EPS, max=1.0)
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * torch.log(torch.clamp(1.0 - probabilities_distance, min=EPS, max=1.0))
        * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE

def umap_loss(embedding_to, embedding_from, min_dist=0.1, batch_size=8, negative_sample_rate=5):
    _a, _b = find_ab_params(1.0, min_dist)
    # grab z for the edge

    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1),
        (embedding_neg_to - embedding_neg_from).norm(dim=1)
    ), dim=0)

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, _a, _b
    )
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
    )

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.cuda(),
        probabilities_distance.cuda(),
    )
    loss = torch.mean(ce_loss)
    return loss

class pumap():
    def __init__(self, model, n_neighbors=15, metric="cosine", random_state=None):
        self.model = model
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.random_state - check_random_state(None) if random_state == None else random_state
    def fit(self, X):
        self.dims = X.shape[1:]
        
        # number of trees in random projection forest
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        # distance metric

        # get nearest neighbors
        nnd = NNDescent(
            X.reshape((len(X), np.product(np.shape(X)[1:]))),
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph

        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph
        # build fuzzy_simplicial_set
        umap_graph, sigmas, rhos = fuzzy_simplicial_set(
            X = X,
            n_neighbors = self.n_neighbors,
            metric = self.metric,
            random_state = self.random_state,
            knn_indices= knn_indices,
            knn_dists = knn_dists,
        )
        
        return umap_graph