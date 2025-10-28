from copyreg import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import numpy as np

from ..models.linear_autoencoder import LinearAutoencoder
from ..models.linear_sparse_autoencoder import LinearSparseAutoencoder
from ..models.denoising_sparse_autoencoder import DenoisingSparseAutoencoder

class MixedManifoldDetector:
    """
    Clase para la representación 2D de patrones de aprendizaje no supervisado.
    
    Esta clase actúa como una "fábrica" que inicializa un modelo de reducción de dimensionalidad específico y lo aplica a los datos.
    """

    def __init__(self, autoencoder=None, manifold_algorithm=TSNE()):
        """
        Inicializa el detector.
        
        Args:
            autoencoder (str): El tipo de modelo a usar ('linear_autoencoder', 'linear_sparse_autoencoder', 'denoising_sparse_autoencoder').
            manifold_algorithm (str): El algoritmo manifold a usar ('lle' y 'tsne').
            **kwargs: Hiperparámetros que se pasarán al modelo (ej. perplexity=30 para t-SNE, random_state=42).
        """
        if autoencoder is None:
            autoencoder = LinearAutoencoder()
        self.autoencoder = autoencoder
        self.manifold_algorithm = manifold_algorithm
        self.embedding_ = None # Aquí se guardará el resultado 2D
        self.is_trained = False

    def fit_transform(self, data):
        """
        Ajusta el modelo a los datos X y devuelve la representación 2D.

        Args:
            X (array-like): Los datos de entrada (features) de forma (n_samples, n_features).

        Returns:
            np.ndarray: La representación 2D (embedding) de forma (n_samples, 2).
        """
        self.autoencoder.fit(data)
        self.is_trained = True
        self.embedding_ = self.autoencoder.transform(data)      
        return self.manifold_algorithm.fit_transform(self.embedding_)  

    def fit(self, data):
        """
        Llama al método fit_transform pero no devuelve nada.
        """
        self.fit_transform(data)

    def transform(self, data, k=5):
        """
        Transforma los datos usando el modelo ajustado.
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a 'fit()' primero.")
        
        embedding = self.autoencoder.transform(data)
        embeddings_trans = []
        for pattern in embedding:
            dist = np.linalg.norm(self.embedding_ - pattern, axis=1)
            # Comprobar si ya existe. Si existe devolver el cálculo anterior
            if np.min(dist) < 1e-9:
                embeddings_trans.append(self.embedding_[np.argmin(dist)])
            else: 
                _, neighbor_indices = NearestNeighbors([pattern], n_neighbors=k)
                embedding_promedio = np.mean(self.embedding_[neighbor_indices[0]], axis=0)
                embeddings_trans.append(embedding_promedio)

        return np.array(embeddings_trans, dtype=np.float32)
