
import numpy as np
import torch
from .autoencoder import Autoencoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LinearAutoencoder(Autoencoder):
    """
    Wrapper para el modelo _InternalAE que implementa la interfaz requerida por MixedManifoldDetector.
    """

    def __init__(self, input_dim:int, **kwargs):
        """
        Constructor con parámetros de entrenamiento.
        """
        super().__init__(**kwargs) # Inicialización del módulo base nn.Module
        
        self.embedding_dim = 32

        # Capas del Encoder (D -> 32)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Capa L1
            nn.ReLU(),
            nn.Linear(256, 128),        # Capa L2
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim) # Capa L3 (Embedding)
        )

        # Capas del Decoder (32 -> D)
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 128), # Capa L4
            nn.ReLU(),
            nn.Linear(128, 256),           # Capa L5
            nn.ReLU(),
            nn.Linear(256, input_dim)      # Capa de salida L6
        )

    def forward(self, x):
        """
        Paso 'forward' completo: codifica y luego decodifica.
        Args:
            x (torch.Tensor): Datos de entrada.
        Returns:
            torch.Tensor: Datos reconstruidos.
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def encode(self, x):
        """
        Método separado para usar solo el encoder (para 'transform').
        Args:
            x (torch.Tensor): Datos de entrada.
            Returns:
            torch.Tensor: Embedding de los datos de entrada.
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Método separado para usar solo el decoder.
        Args:
            z (torch.Tensor): Embedding de los datos.
        Returns:
            torch.Tensor: Datos reconstruidos a partir del embedding.
        """
        return self.decoder(z)
    