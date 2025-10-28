import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .autoencoder import Autoencoder


class DenoisingSparseAutoencoder(Autoencoder):
    """
    Implementa un Autoencoder Lineal Sparse y Denoising.
    Hereda de Autoencoder y sobreescribe el método 'fit' para: añadir ruido Gaussiano a la entrada y añadir una penalización L1 (Sparsity) a la función de pérdida.
    """

    def __init__(self, input_dim: int, **kwargs):
        """
        Constructor con parámetros de entrenamiento.
        """
        super().__init__(**kwargs) 
        
        self.embedding_dim = 32

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Capa L1
            nn.ReLU(),
            nn.Linear(256, 128),        # Capa L2
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim) # Capa L3 (Embedding)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 128), # Capa L4
            nn.ReLU(),
            nn.Linear(128, 256),           # Capa L5
            nn.ReLU(),
            nn.Linear(256, input_dim)      # Capa de salida L6
        )

        self.to(self.device)

    def fit(self, data: np.ndarray):
        """
        Recibe datos NumPy y entrena el modelo de PyTorch. (Modificado para 'denoising' y 'sparsity')
        """
         # Configurar pérdida y optimizador
        criterion = nn.MSELoss() # Error Cuadrático Medio
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Configurar datos (convertir NumPy a Tensors de PyTorch)
        dataset = TensorDataset(torch.tensor(data.values, dtype=torch.float32, device=self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train()

        # Bucle de entrenamiento
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_data in loader:
                batch = batch_data[0].to(self.device)
                
                corrupted_batch = batch + self.noise_weight * torch.randn_like(batch)
                
                # Forward pass
                encoded = self.encoder(corrupted_batch)
                reconstructed = self.decoder(encoded)
                
                # Calcular pérdida MSE
                mse_loss = criterion(reconstructed, batch)
                l1_penalty = self.sparsity_weight * torch.sum(torch.abs(encoded))
                loss = mse_loss + l1_penalty
                                
                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calcular pérdida promedio de la época
            avg_loss = total_loss / len(loader)
            
            # Mostrar progreso cada 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Pérdida (MSE+L1+Ruido): {avg_loss:.6f}')

            # Comprobar umbral de error para parada temprana
            if avg_loss <= self.error_threshold:
                print(f'Umbral de error alcanzado. Deteniendo entrenamiento en epoch {epoch+1}.')
                break

            self.is_trained = True
    
    def forward(self, x):
        """
        Paso 'forward' completo: codifica y luego decodifica.
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def encode(self, x):
        """
        Método separado para usar solo el encoder (para 'transform').
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Método separado para usar solo el decoder.
        """
        return self.decoder(z)
    