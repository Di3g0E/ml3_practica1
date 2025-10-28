import os
import sys
import pandas as pd
    
def save_experiment(csv_path, experiment_name,
                    trustworthiness_train, trustworthiness_test,
                    train_time, test_time):
    """Guarda los resultados del experimento en un CSV y las embeddings en un archivo pickle.
    
    Args:
        csv_path (str): Ruta al archivo CSV donde se guardarán los resultados.
        experiment_name (str): Nombre del experimento.
        trustworthiness_train (float): Trustworthiness de los datos de entrenamiento.
        trustworthiness_test (float): Trustworthiness de los datos de test.
        fit_time (float): Tiempo de ajuste del modelo.
        train_time (float): Tiempo de transformación de los datos de entrenamiento.
        test_time (float): Tiempo de transformación de los datos de test.
    """

    results = {
        'experiment_name': experiment_name,
        'train_time': train_time,
        'test_time': test_time,
        'trustworthiness_train': trustworthiness_train,
        'trustworthiness_test': trustworthiness_test
    }
    df = pd.DataFrame([results])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing.dropna(how='all')
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)

def load_dataset_csv(path, train="mnist_train.csv", test=None, sample_fraction=None):
    """Carga un dataset desde un archivo CSV.
    
    Args:
        path (str): Ruta al archivo CSV.
        train (str): Nombre del archivo CSV de entrenamiento.
        test (str): Nombre del archivo CSV de prueba (opcional).
        
    Returns:
        tuple: (labels, features) donde labels es un array de etiquetas y features es un array de características.
    """
    # Si no existe el archivo, lanzar error
    if not os.path.exists('./data/' + train):
        print(f"El archivo {train} no existe en la carpeta './data/'.")
        sys.exit(1)
        
    train_df = pd.read_csv('./data/' + train)
    if test:
        test_df = pd.read_csv('./data/' + test)

    if sample_fraction is not None and 0 < sample_fraction < 1:
        train_df = train_df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        if test:
            test_df = test_df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    return train_df, test_df if test else None
    