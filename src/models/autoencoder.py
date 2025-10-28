import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class Autoencoder(nn.Module):
    """
    Clase base abstracta (hereda de nn.Module) que define la interfaz y el bucle de entrenamiento genérico (`fit`) para todos los modelos de autoencoder del proyecto.
    """
    def __init__(self, epochs=100, error_threshold=0.001, batch_size=32, lr=0.001, embedding_dim=32, sparsity_weight=1e-3, noise_weight=0.3):
        super(Autoencoder, self).__init__() # Inicialización del módulo base nn.Module
        
        self.epochs = epochs
        self.error_threshold = error_threshold
        self.batch_size = batch_size
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.sparsity_weight = sparsity_weight
        self.noise_weight = noise_weight

        # Selecciona GPU si está disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_trained = False

    def fit(self, data: np.ndarray):
        """
        Recibe datos NumPy y entrena el modelo de PyTorch.
        Args:
            data (np.ndarray): Datos de entrenamiento en formato NumPy.
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
                
                # Forward pass
                reconstructed = self(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calcular pérdida promedio de la época
            avg_loss = total_loss / len(loader)
            
            # Mostrar progreso cada 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Pérdida (MSE): {avg_loss:.6f}')

            # Comprobar umbral de error para parada temprana
            if avg_loss <= self.error_threshold:
                print(f'Umbral de error alcanzado. Deteniendo entrenamiento en epoch {epoch+1}.')
                break

            self.is_trained = True

    def transform(self, data):
        """
        Recibe datos NumPy y devuelve el embedding de 32 dimensiones.
        Args:
            data (np.ndarray): Datos de entrada en formato NumPy.
        Returns:
            np.ndarray: Embedding resultante en formato NumPy.
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a 'fit()' primero.")
        
        # Poner el modelo en modo evaluación (desactiva dropout, etc.)
        self.eval()
        
        with torch.no_grad():
            data_tensor = torch.tensor(data.values, dtype=torch.float32, device=self.device)
            embeddings = self.encoder(data_tensor)
            
        return embeddings.cpu().numpy()
