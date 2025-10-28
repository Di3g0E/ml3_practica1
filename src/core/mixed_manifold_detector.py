from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from copyreg import pickle
import numpy as np

from ..models.linear_autoencoder import LinearAutoencoder

class MixedManifoldDetector:
    """
    Clase para la representación 2D de patrones de aprendizaje no supervisado.
    Esta clase actúa como una "fábrica" que inicializa un modelo de reducción de dimensionalidad específico y lo aplica a los datos.
    """

    def __init__(self, autoencoder=None, manifold_algorithm=TSNE()):
        """
        Inicializa el detector.
        """
        if autoencoder is None:
            autoencoder = LinearAutoencoder(input_dim=785)
            
        self.autoencoder = autoencoder
        self.manifold_algorithm = manifold_algorithm
        
        self.embedding_ = None 
        self.embedding_2d_ = None 
        self.is_trained = False

    def fit_transform(self, data):
        """
        Ajusta el modelo a los datos y devuelve la representación 2D.
        """
        self.autoencoder.fit(data)
        self.is_trained = True
        
        self.embedding_ = self.autoencoder.transform(data)
        self.embedding_2d_ = self.manifold_algorithm.fit_transform(self.embedding_)
        return self.embedding_2d_

    def fit(self, data):
        """
        Llama al método fit_transform pero no devuelve nada.
        """
        self.fit_transform(data)

    def transform(self, data, k=5):
        """
        Transforma los datos usando el modelo ajustado.
        Encuentra los k-vecinos más cercanos en el espacio latente (32D) y devuelve el promedio de sus posiciones en el espacio 2D.
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a 'fit()' primero.")
        
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.embedding_)

        embedding_32d_new = self.autoencoder.transform(data)
        embeddings_2d_trans = []
        
        for pattern_32d in embedding_32d_new:
            neighbor_indices = nn.kneighbors([pattern_32d], return_distance=False)
            corresponding_2d_neighbors = self.embedding_2d_[neighbor_indices[0]]

            embedding_promedio_2d = np.mean(corresponding_2d_neighbors, axis=0)
            embeddings_2d_trans.append(embedding_promedio_2d)

        return np.array(embeddings_2d_trans, dtype=np.float32)
    