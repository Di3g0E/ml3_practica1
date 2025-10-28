import sys
import os
import numpy as np
import torch

# --- Configuración de ruta con Gemini 2.5 Pro ---
# Esto es crucial para que el script 'test' encuentre la carpeta 'src'
#
# 1. Obtener la ruta absoluta de este archivo (test_pca.py)
#    ej: /.../tu_proyecto/test
current_dir = os.path.dirname(os.path.abspath(__file__))
#
# 2. Subir un nivel para llegar a la raíz del proyecto
#    ej: /.../tu_proyecto/
project_root = os.path.abspath(os.path.join(current_dir, '..'))
#
# 3. Añadir la raíz del proyecto al path de Python
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#
# ¡Ahora podemos importar desde 'src' con normalidad!
# ------------------------------

try:
    from src.core.mixed_manifold_detector import MixedManifoldDetector

except ImportError as e:
    print(f"Error: No se pudo importar DenoisingSparseAutoencoder desde src.models.denoising_sparse_autoencoder")
    print(f"Asegúrate de que la estructura de carpetas es correcta.")
    print(f"Detalle del error: {e}")
    sys.exit(1)

# Función de prueba para DenoisingSparseAutoencoder con el df de entrenamiento cargado
def test_denoising_sparse_autoencoder_model(train_df):
    """
    Prueba el funcionamiento básico de DenoisingSparseAutoencoder.
    """
    # 1. ARRANGE (Preparar): Crear datos de prueba
    n_samples = train_df.shape[0]
    n_features = train_df.shape[1]

    # Parametros de prueba
    torch.manual_seed(42)
    np.random.seed(42)
    embedding_dim = 32
    test_epochs = 5
    test_batch_size = 16
    lr = 0.001
    error_threshold = 0.001

    ae = MixedManifoldDetector(autoencoder='denoising_sparse_autoencoder', epochs=test_epochs, batch_size=test_batch_size, lr=lr, error_threshold=error_threshold, sparsity_weight=1e-3, corruption_level=0.1)
    
    # Entrenar el modelo
    ae.fit(train_df)
    
    # Obtener el embedding (la capa de 32 neuronas)
    embedding = ae.transform(train_df)

    # 3. ASSERT (Verificar) - Prueba 1
    assert embedding is not None, "El embedding no debería ser None."
    
    # Comprobar que la salida es un array de NumPy
    assert isinstance(embedding, np.ndarray), \
        f"El tipo de salida debería ser np.ndarray, pero fue {type(embedding)}"
    
    # La forma resultante debe ser (n_muestras, dimension_embedding)
    expected_shape = (n_samples, embedding_dim)
    assert embedding.shape == expected_shape, \
        f"Error en la forma (shape) del embedding. " \
        f"Se esperaba {expected_shape} pero se obtuvo {embedding.shape}"

    print("Test de DenoisingSparseAutoencoder pasado correctamente.\n")
