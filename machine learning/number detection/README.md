# MNIST Digit Recognition Project 📊

Este proyecto utiliza técnicas avanzadas de procesamiento de imágenes y machine learning para reconocer dígitos del conjunto de datos MNIST. Incluye funcionalidades para entrenamiento de modelos, pruebas, y una interfaz de usuario interactiva para demostración.
Descripción del Problema

El reconocimiento de dígitos es una rama del procesamiento de imágenes y aprendizaje automático que se ha convertido en una herramienta esencial, no solo para iniciarse en el mundo del aprendizaje automático, sino también para su aplicación en numerosas prácticas y en la investigación de tecnologías avanzadas. Investigaciones como el estudio "Reconocimiento de dígitos escritos a mano mediante métodos de tratamiento de imagen y modelos de clasificación" (Luis Miralles Pechuán, 2015), han demostrado que el reconocimiento óptico de caracteres (OCR) se extiende más allá de la digitalización de documentos textuales para incluir la interpretación de gestos y movimientos en el espacio físico, ampliando significativamente su aplicabilidad.
Solución Propuesta

La propuesta para el reconocimiento de dígitos trazados en el aire se centra en el uso de un modelo especializado de Máquina de Vectores de Soporte (SVM), diseñado para clasificar eficazmente dígitos bajo diversas condiciones. Esta solución se apoya en la robustez de las SVM para adaptarse a la diversidad en la forma de los dígitos generados por distintos gestos y variaciones en la iluminación. Se empleará visión por computadora para capturar y analizar en tiempo real los dígitos dibujados, utilizando técnicas de detección de bordes para diferenciar los dígitos del entorno.
Estructura del Proyecto 📁

    models/: Modelos de clasificación y escaladores.
    test/: Scripts y notebooks para pruebas de los modelos, incluyendo predicciones, matrices de confusión, y métricas de rendimiento.
    train/: Scripts para entrenar los modelos utilizando tanto técnicas estándares como PCA.
    view/: Scripts de la interfaz de usuario que permiten interactuar con los modelos a través de una interfaz gráfica.
    venv/: Entorno virtual para manejar las dependencias.
    requirements.txt: Dependencias del proyecto necesarias para su ejecución.

Componentes Principales 🔑
Modelos de Aprendizaje Automático 🧠

    train_model.py y train_model_pca.py: Entrenan modelos con y sin PCA.
    hand_detection.py, hand_drawing.py, y number_detection.py: Permiten la interacción con el sistema mediante detección y dibujo de manos, y reconocimiento de números.
    main.py: Ejecuta la interfaz principal con botones para las diferentes funcionalidades.

Tecnologías y Bibliotecas Utilizadas 🛠️

    OpenCV (cv2): Para el manejo de imágenes y operaciones de visión por computadora.
    MediaPipe: Detecta estructuras de la mano en tiempo real.
    NumPy: Gestiona la manipulación de datos en formato de matrices o arrays.
    scikit-learn: Utilizado para implementar algoritmos de machine learning, especialmente SVM para la clasificación de dígitos.
    joblib: Para cargar y guardar los modelos entrenados.
    Tkinter: Para desarrollar la interfaz gráfica de usuario.

Cómo Empezar 🚀
Clonar el Repositorio

bash

    git clone
    cd Proyecto

Configurar el Entorno Virtual

bash

    python -m venv venv
    source venv/bin/activate  # En Windows use `venv\Scripts\activate`

Instalar Dependencias

bash

    pip install -r requirements.txt

Ejecutar la Interfaz Principal

bash

    python view/main.py

Pruebas 🧪

Las pruebas se pueden realizar utilizando el archivo test.ipynb, disponible en la carpeta test/. Este notebook puede ser ejecutado localmente en Jupyter Notebook o en Google Colab para visualizar los resultados, realizar predicciones, y analizar las métricas de rendimiento de los modelos entrenados.
Licencia 📄

Este proyecto está bajo la Licencia MIT, que permite el uso, copia, modificación, fusión, publicación, distribución, sublicencia, y/o venta de copias del software, y permite a las personas a quienes se les proporcione el software hacer lo mismo, siempre que se incluya el aviso de derechos de autor y las condiciones de la licencia en todas las copias o partes sustanciales del software.

Proyecto realizado en conjunto de mis compañeros Daniela Pinzon, Miguel Thomas y Daniel Oviedo
