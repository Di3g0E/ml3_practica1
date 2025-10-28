
import numpy as np
import torch
from .autoencoder import Autoencoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LinearSparseAutoencoder(Autoencoder):
    """
    Wrapper para el modelo _InternalAE que implementa la interfaz requerida por MixedManifoldDetector.
    """

    def __init__(self, epochs=100, lr=0.001, error_threshold=0.001, batch_size=32, sparsity_weight=1e-3):
        """
        Constructor con parámetros de entrenamiento.
        Args:
            epochs (int): Número de epochs para el entrenamiento.
            lr (float): Learning rate para el optimizador.
            error_threshold (float): Umbral de error para detener el entrenamiento.
            batch_size (int): Tamaño de los batches para el DataLoader.
            sparsity_weight (float): Peso de la regularización L1 (sparse).
        """
        super(LinearSparseAutoencoder, self).__init__() # Inicialización del módulo base nn.Module

        self.epochs = epochs
        self.lr = lr
        self.error_threshold = error_threshold
        self.batch_size = batch_size
        self.sparsity_weight = sparsity_weight
        
        # Selecciona GPU si está disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, data: np.ndarray):
        """
        Recibe datos NumPy y entrena el modelo de PyTorch.
        Args:
            data (np.ndarray): Datos de entrenamiento en formato NumPy.
        """
        if data.ndim != 2:
            raise ValueError(f"Los datos de entrada deben ser 2D (samples, features), "
                             f"pero se recibió {data.shape}")
        
        n_samples, input_dim = data.shape
        print(f"Iniciando 'fit' en datos con forma: ({n_samples}, {input_dim})")
        
        # 1. Inicializar el modelo interno y moverlo al dispositivo
        self.model = Autoencoder(input_dim=input_dim, embedding_dim=32).to(self.device)
        self.model.train() # Poner el modelo en modo entrenamiento

        # 2. Configurar datos (convertir NumPy a Tensors de PyTorch)
        data_array = data.values.astype(np.float32) # Convierte el DataFrame a numpy porque PyTorch no acepta DataFrames directamente como tensor
        dataset = TensorDataset(torch.tensor(data_array, dtype=torch.float32, device=self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. Configurar pérdida y optimizador
        criterion = nn.MSELoss() # Error Cuadrático Medio
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 4. Bucle de entrenamiento
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_data in loader:
                # El loader devuelve una lista, tomamos el primer (y único) tensor
                batch = batch_data[0].to(self.device)
                
                # Forward pass
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += (loss.item() + self.sparsity_weight * torch.sum(torch.abs(self.model.encode(batch))).item()) # Agregar término de regularización L1
            
            # Calcular pérdida promedio de la época
            avg_loss = total_loss / len(loader)
            
            # Mostrar progreso cada 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Pérdida (MSE): {avg_loss:.6f}')

            # Comprobar umbral de error para parada temprana
            if avg_loss <= self.error_threshold:
                print(f'Umbral de error alcanzado. Deteniendo entrenamiento en epoch {epoch+1}.')
                break
                
    
    