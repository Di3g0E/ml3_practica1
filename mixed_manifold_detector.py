import argparse
import os
import time
from sklearn.manifold import trustworthiness, TSNE

from src.core.mixed_manifold_detector import MixedManifoldDetector
from src.utils.io import save_experiment, load_dataset_csv
from src.models.linear_autoencoder import LinearAutoencoder

def main():
    parser = argparse.ArgumentParser(description="MixedManifoldDetector CLI")
    parser.add_argument("train_file", type=str, help="Nombre del archivo CSV de entrenamiento (ej: mnist_train.csv)")
    parser.add_argument("test_file", type=str, nargs="?", help="Nombre del archivo CSV de test (ej: mnist_test.csv)")
    args = parser.parse_args()

    # Cargamos los dataframes con opcion de coger solo una muestra del 10%
    train_df, test_df = load_dataset_csv('./data/', train=args.train_file, test=args.test_file, sample_fraction=0.1)
    
    print(train_df.shape, test_df.shape)

    # Directorios para guardar resultados
    results_subpath = os.path.join("artifacts") # Directorio para guardar los resultados
    csv_path = os.path.join(results_subpath, "metrics", "metrics_output.csv") # Directorio donde se guarda el CSV con los resultados

    # Crear instancias
    autoencoder = LinearAutoencoder(input_dim=train_df.shape[1], epochs=5)
    manifold_algorithm = TSNE(n_components=2)

    detector = MixedManifoldDetector(autoencoder=autoencoder, manifold_algorithm=manifold_algorithm)
    
    # Entrenamos el detector
    start_train = time.time()
    train_embedding = detector.fit_transform(train_df)
    train_time = time.time() - start_train

    if args.test_file:
        start_test = time.time()
        test_embedding = detector.fit_transform(test_df)
        test_time = time.time - start_test

    # Calcular trustworthiness para train y test
    trustworthiness_train = trustworthiness(train_df, train_embedding)
    trustworthiness_test = trustworthiness(test_df, test_embedding)

    # Guardar datos
    save_experiment(csv_path, args.train_file.split('.')[0]+"_experiment",
                    trustworthiness_train, trustworthiness_test,
                    train_time, test_time)

if __name__ == "__main__":
    main()
