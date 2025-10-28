import sys
import os

# --- INICIO: Corrección de Ruta por Gemini---
# 1. Obtener la ruta absoluta de este script (que está en 'test/')
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subir un nivel para llegar a la raíz del proyecto
#    (ej: de '.../tu_proyecto/test' a '.../tu_proyecto/')
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 3. Añadir la raíz del proyecto al path de Python
#    Ahora Python podrá encontrar la carpeta 'src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. Cambiar el directorio de trabajo actual a la raíz del proyecto
#    Esto es CRUCIAL para que './data/' y 'artifacts/' funcionen.
os.chdir(project_root)
# --- FIN: Corrección de Ruta por Gemini ---

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness, TSNE, LocallyLinearEmbedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

from src.core.mixed_manifold_detector import MixedManifoldDetector
from src.utils.io import load_dataset_csv 
from src.models.linear_autoencoder import LinearAutoencoder
from src.models.linear_sparse_autoencoder import LinearSparseAutoencoder
from src.models.denoising_sparse_autoencoder import DenoisingSparseAutoencoder

# --- Configuración Global de Experimentos ---
SAMPLE_FRACTION = 0.5
EPOCHS = 50

PLOTS_DIR = os.path.join("artifacts", "plots")
METRICS_DIR = os.path.join("artifacts", "metrics")
CSV_PATH = os.path.join(METRICS_DIR, "full_experiment_results.csv")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def separate_labels(df, label_col_name=None):
    """
    Separa features (X) y etiquetas (y) de un DataFrame.
    Si label_col_name no se da, asume que es la PRIMERA columna.
    """
    if label_col_name:
        y = df[label_col_name]
        X = df.drop(columns=[label_col_name])
    else:
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    return X, y

def load_csv_dataset_xy(train_file, test_file, sample_fraction=1.0):
    """
    Carga train/test de CSVs, separa X/y, y aplica sample_fraction.
    """
    print(f"Cargando dataset desde CSVs: {train_file}, {test_file}...")
    
    train_df, test_df = load_dataset_csv(
        train=train_file, 
        test=test_file, 
        sample_fraction=sample_fraction
    )
    
    X_train, y_train = separate_labels(train_df)
    X_test, y_test = separate_labels(test_df)

    print(f"{train_file} -> Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def calculate_distance_correlation(X_orig, X_embed):
    """
    Calcula la correlación de Spearman en las distancias por pares. Función dada por Gemini
    """
    try:
        dist_orig = pairwise_distances(X_orig.values if isinstance(X_orig, pd.DataFrame) else X_orig)
        dist_embed = pairwise_distances(X_embed)
        
        triu_indices = np.triu_indices_from(dist_orig, k=1)
        dist_orig_flat = dist_orig[triu_indices]
        dist_embed_flat = dist_embed[triu_indices]
        
        corr, _ = spearmanr(dist_orig_flat, dist_embed_flat)
        return corr
    except Exception as e:
        print(f"  Advertencia: No se pudo calcular la correlación de distancias: {e}")
        return np.nan


def plot_embedding(embedding, labels, title, filepath):
    """
    Guarda un gráfico de dispersión del embedding coloreado por etiquetas. Función dada por Gemini
    """
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Paired', s=5, alpha=0.7)
        plt.title(title)
        plt.savefig(filepath)
        plt.close()
    except Exception as e:
        print(f"  Advertencia: No se pudo guardar el gráfico en {filepath}: {e}")


def main():
    """
    Función dada por Gemini y modificada por Diego Esclarín.
    """
    datasets = [
        {"name": "mnist", "train_file": "mnist_train.csv", "test_file": "mnist_test.csv"},
        {"name": "fashion_mnist", "train_file": "fashion_mnist_train.csv", "test_file": "fashion_mnist_test.csv"},
        {"name": "cifar10", "train_file": "cifar10_train.csv", "test_file": "cifar10_test.csv"}
    ]

    autoencoders = [
        {"name": "LinearAE", "class": LinearAutoencoder, "kwargs": {}},
        {"name": "LinearSparseAE", "class": LinearSparseAutoencoder, "kwargs": {"sparsity_weight": 1e-5}},
        {"name": "DenoisingSparseAE", "class": DenoisingSparseAutoencoder, "kwargs": {"sparsity_weight": 1e-5, "noise_weight": 0.1}}
    ]

    manifolds = [
        {"name": "TSNE", "class": TSNE, "kwargs": {"n_components": 2, "random_state": 42, "init": "pca", "learning_rate": "auto"}},
        {"name": "LLE", "class": LocallyLinearEmbedding, "kwargs": {"n_components": 2, "random_state": 42}}
    ]

    # Bucle de Pruebas
    
    print(f"--- Iniciando Batería de Pruebas ---")
    print(f"Directorio de trabajo: {os.getcwd()}")
    print(f"Epochs por modelo: {EPOCHS}")
    print(f"Fracción de datos: {SAMPLE_FRACTION*100}%")
    print(f"Resultados en: {CSV_PATH}")
    print(f"Gráficos en: {PLOTS_DIR}\n")
    
    total_experiments = len(datasets) * len(autoencoders) * len(manifolds)
    exp_count = 0
    all_results = [] 

    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        
        try:
            X_train, y_train, X_test, y_test = load_csv_dataset_xy(
                dataset_config["train_file"], 
                dataset_config["test_file"], 
                sample_fraction=SAMPLE_FRACTION
            )
        except Exception as e:
            print(f"ERROR: No se pudo cargar el dataset {dataset_name}. Saltando... (Detalle: {e})")
            exp_count += len(autoencoders) * len(manifolds)
            continue
            
        input_dim = X_train.shape[1]

        for ae_config in autoencoders:
            ae_name = ae_config["name"]
            
            ae_kwargs = ae_config["kwargs"].copy()
            ae_kwargs['input_dim'] = input_dim
            ae_kwargs['epochs'] = EPOCHS
            
            for manifold_config in manifolds:
                manifold_name = manifold_config["name"]
                exp_count += 1
                
                experiment_name = f"{dataset_name}_{ae_name}_{manifold_name}"
                print(f"--- ({exp_count}/{total_experiments}) Ejecutando: {experiment_name} ---")

                try:
                    # 1. Instanciar
                    autoencoder = ae_config["class"](**ae_kwargs)
                    manifold = manifold_config["class"](**manifold_config["kwargs"])
                    detector = MixedManifoldDetector(autoencoder=autoencoder, manifold_algorithm=manifold)

                    # 2. Entrenar
                    print(f"  Ajustando en X_train ({X_train.shape})...")
                    start_train = time.time()
                    train_embedding = detector.fit_transform(X_train) 
                    fit_time = time.time() - start_train

                    # 3. Evaluar
                    print(f"  Transformando X_test ({X_test.shape})...")
                    start_test = time.time()
                    test_embedding = detector.transform(X_test)
                    transform_time = time.time() - start_test
                    
                    # 4. Métricas Cuantitativas
                    print("  Calculando métricas...")
                    trust_train = trustworthiness(X_train.values, train_embedding, n_neighbors=12)
                    trust_test = trustworthiness(X_test.values, test_embedding, n_neighbors=12)
                    
                    knn = KNeighborsClassifier(n_neighbors=7)
                    knn.fit(train_embedding, y_train)
                    knn_accuracy = knn.score(test_embedding, y_test)
                    
                    dist_corr = calculate_distance_correlation(X_test, test_embedding)

                    # 5. Gráficos (Evaluación Cualitativa)
                    print("  Guardando gráficos...")
                    plot_filepath_train = os.path.join(PLOTS_DIR, f"{experiment_name}_train.png")
                    plot_filepath_test = os.path.join(PLOTS_DIR, f"{experiment_name}_test.png")
                    
                    plot_embedding(train_embedding, y_train, f"{experiment_name} - Train", plot_filepath_train)
                    plot_embedding(test_embedding, y_test, f"{experiment_name} - Test", plot_filepath_test)

                    # 6. Almacenar resultados
                    result_entry = {
                        "experiment_name": experiment_name,
                        "dataset": dataset_name,
                        "autoencoder": ae_name,
                        "manifold": manifold_name,
                        "fit_time": fit_time,
                        "transform_time": transform_time,
                        "trustworthiness_train": trust_train,
                        "trustworthiness_test": trust_test,
                        "knn_accuracy": knn_accuracy,
                        "distance_correlation": dist_corr,
                        "plot_train": f"{experiment_name}_train.png",
                        "plot_test": f"{experiment_name}_test.png"
                    }
                    all_results.append(result_entry)
                    
                    print(f"  Éxito. k-NN Acc: {knn_accuracy:.4f}, Trust (Test): {trust_test:.4f}, DistCorr: {dist_corr:.4f}\n")

                except Exception as e:
                    print(f"ERROR durante el experimento {experiment_name}. Saltando...")
                    print(f"Detalle del error: {e}\n")

    # --- 5. Guardar CSV ---
    if all_results:
        print(f"Guardando {len(all_results)} resultados en {CSV_PATH}...")
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(CSV_PATH, index=False, sep=';')
    
    print("--- Batería de Pruebas Completada ---")

if __name__ == "__main__":
    main()
