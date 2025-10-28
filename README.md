# ml3_practica1: Detector de Manifolds Mixto (Mixed Manifold Detector)

Este proyecto implementa un `MixedManifoldDetector`, una herramienta diseñada para la visualización de datos de alta dimensionalidad en un espacio 2D. El sistema utiliza un enfoque de dos etapas:

1.  **Reducción de Dimensión (Autoencoder):** Primero, un Autoencoder (AE) basado en PyTorch comprime los datos de entrada en un espacio latente de dimensiones reducidas (p. ej., 32 dimensiones).
2.  **Visualización (Manifold):** Luego, un algoritmo de aprendizaje de variedades (manifold learning) de scikit-learn (como TSNE o LLE) se aplica a ese espacio latente para reducirlo aún más a 2 dimensiones, permitiendo su visualización en un gráfico de dispersión.

## 1. Arquitectura del Proyecto

La estructura del repositorio está organizada de la siguiente manera:

* `/src/`: Contiene toda la lógica fuente del detector.
    * `/src/models/`: Define la clase base abstracta `Autoencoder` y sus tres implementaciones concretas (Linear, Sparse, Denoising).
    * `/src/core/`: Incluye la clase principal `MixedManifoldDetector`, que encapsula y combina el autoencoder y el algoritmo de manifold.
    * `/src/utils/`: Contiene utilidades, como `io.py` para cargar datasets y guardar métricas.
* `/data/`: Directorio destinado a alojar los archivos `.csv` de los datasets (mnist, fashion_mnist, cifar10).
* `/test/`: Contiene los scripts para ejecutar las baterías de pruebas y experimentos.
* `/artifacts/`: Directorio donde se almacenan todos los resultados generados.
    * `/artifacts/metrics/`: Almacena los resultados cuantitativos en formato `.csv`.
    * `/artifacts/plots/`: Almacena las visualizaciones `.png` de los embeddings 2D.
* `/docs/`: Contiene diagramas de arquitectura del proyecto.
* `mixed_manifold_detector.py`: Un script CLI principal para ejecuciones rápidas y pruebas.
* `environment.yml`: Archivo de entorno de Conda para replicar las dependencias.

## 2. Características Implementadas

### Modelos de Autoencoder

Se han implementado tres variantes de Autoencoders, todas heredando de una clase base común `Autoencoder`:

1.  **`LinearAutoencoder`:** Un autoencoder denso estándar.
2.  **`LinearSparseAutoencoder`:** Añade regularización L1 (sparsity) a la función de pérdida sobre el espacio latente, para fomentar representaciones dispersas.
3.  **`DenoisingSparseAutoencoder`:** Añade ruido gaussiano a la entrada del modelo durante el entrenamiento, además de la regularización L1, para aprender representaciones más robustas.

### Algoritmos Manifold

El detector se ha probado con dos algoritmos de reducción de dimensionalidad de `scikit-learn`:

1.  **`TSNE`** (t-distributed Stochastic Neighbor Embedding)
2.  **`LLE`** (Locally Linear Embedding)

## 3. Instalación

Para configurar el entorno de desarrollo, sigue estos pasos:

1.  Clona el repositorio en tu máquina local.
2.  Asegúrate de tener Anaconda o Miniconda instalado.
3.  Crea y activa el entorno de Conda usando el archivo `environment.yml` proporcionado:

    ```bash
    # Crear el entorno
    conda env create -f environment.yml
    
    # Activar el entorno
    conda activate ml3p1env
    ```

## 4. Guía de Ejecución

### a. Preparación de Datos

Antes de ejecutar cualquier script, debes **colocar los archivos `.csv` de los datasets** (ej: `mnist_train.csv`, `mnist_test.csv`, `cifar10_train.csv`, etc.) en el directorio `./data/`.

### b. Ejecución Rápida (CLI)

El script `mixed_manifold_detector.py` en la raíz del proyecto sirve como un punto de entrada rápido. Por defecto, utiliza el `DenoisingSparseAutoencoder` y `TSNE`.

```bash
# Sintaxis
python mixed_manifold_detector.py <archivo_train.csv> [archivo_test.csv]

# Ejemplo con MNIST
python mixed_manifold_detector.py mnist_train.csv mnist_test.csv
