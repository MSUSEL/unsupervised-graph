# coding:utf-8
import numpy as np
import networkx as nx
from typing import List
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

from ksvd import ApproximateKSVD

from collections import Counter

class WL_KSVD(Estimator):
    r""" An implementation of WL_KSVD

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        erase_base_features (bool): Erasing the base features. Default is False.

        n_vocab: Number of preliminary vocabulary size.  Default is 1000
        n_atoms: Number of dictionary elements (atoms). Default is 128
        n_nonzero_coefs: Number of nonzero coefficients to target. Default is 10
        max_iter: Maximum number of iterations. Default is 10
        tol: Tolerance for error. Default is 1e-6

    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
        n_vocab: int = 1000,
        n_atoms: int = 128,
        n_non_zero_coefs: int = 10,
        max_iter: int = 10,
        tol: float = 1e-6

    ):
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.n_atoms = n_atoms
        self.n_non_zero_coefs = n_non_zero_coefs
        self.max_iter = max_iter
        self.tol = tol


    def createWLhash(self, graph_list):

        documents = []
        # TODO: parallel implementation
        for graph in graph_list:
            g = self._check_graph(graph)

            document = WeisfeilerLehmanHashing(
                g, self.wl_iterations, self.attributed, self.erase_base_features)

            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

    def create_vocab(self, corpus):
        d2v_model = Doc2Vec(vector_size=self.n_vocab, min_count=self.min_count, epochs=self.epochs)

        # d2v_model.build_vocab(train_corpus)
        total_words, corpus_count = d2v_model.scan_vocab(
            corpus_iterable=corpus, corpus_file=None,
            progress_per=10000, trim_rule=None
        )
        d2v_model.corpus_count = corpus_count
        d2v_model.corpus_total_words = total_words
        d2v_model.prepare_vocab(update=False, keep_raw_vocab=True, trim_rule=None)

        sorted_vocab = (sorted(d2v_model.raw_vocab.items(), key=lambda item: item[1], reverse=True))

        trimmed_vocab = sorted_vocab[0:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, corpus, vocab):

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        i = 0
        for corpus in corpus:
            words = corpus.words

            words_count = Counter(corpus.words)
            j = 0
            for atom, _ in vocab:
                sparse_vector[i][j] = words_count[atom]
                j = j + 1

            i = i + 1

        return sparse_vector

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a WL_KSVD model.
        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()

        documents = self.createWLhash(graphs)

        self._vocab = self.create_vocab(documents)

        x = self.calc_coefficients(documents, self._vocab)

        aksvd = ApproximateKSVD(n_components=self.dimensions, max_iter=self.max_iter, tol=self.tol,
                 transform_n_nonzero_coefs=self.n_non_zero_coefs)
        self._dictionary = aksvd.fit(x).components_

        self._embedding = aksvd.transform(x)

        self.aksvd = aksvd

        return self

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Infer the graph embeddings.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()

        documents = self.createWLhash(graphs)

        x = self.calc_coefficients(documents, self._vocab)

        embedding = self.aksvd.transform(x)

        return embedding
