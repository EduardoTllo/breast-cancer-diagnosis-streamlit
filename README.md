# 🎗️ Sistema de Diagnóstico Asistido por IA (Breast Cancer Diagnosis Streamlit App)

**[👉 Prueba la aplicación en vivo aquí 👈](https://breast-cancer-prediction-project3.streamlit.app/)**

Esta es una aplicación web interactiva desarrollada con [Streamlit](https://streamlit.io/) para la clasificación de imágenes histopatológicas de cáncer de mama. La aplicación utiliza un modelo **ResNet18** pre-entrenado para predecir si el tejido tumoral presenta características **Benignas** o **Malignas**, proporcionando mapas de calor visuales (Grad-CAM) para explicar las decisiones del modelo.

> **Importante:** Este repositorio contiene únicamente el código de la aplicación de usuario (Front-end) y el script de inferencia del modelo. 
> 🔗 **Todo el trabajo de investigación de datos, experimentación de arquitecturas y entrenamiento de los modelos originales lo puedes encontrar en el repositorio principal del proyecto:** 
> **[Vasco2510/Cancer-detection-with-CNN---High-performance](https://github.com/Vasco2510/Cancer-detection-with-CNN---High-performance)**

---

## 🚀 Características y Funcionalidades Principales

*   **Inferencia mediante ResNet18**: Utiliza pesos de un modelo entrenado específicamente para optimizar la pérdida focal (*focal loss*) y equilibrar predicciones en clases médicas complejas.
*   **Diagnóstico a Nivel de Paciente (Sistema de Votación)**: Permite subir múltiples imágenes del tejido de un mismo paciente. El sistema realiza una evaluación de forma individual y posteriormente combina las probabilidades para entregar un único diagnóstico global más robusto.
*   **Interpretabilidad Visual (Grad-CAM)**: Genera y superpone de forma automática mapas de calor sobre las imágenes histológicas, destacando las regiones celulares que la red neuronal consideró críticas para identificar un tejido como maligno o benigno.
*   **Normalización de Macenko (Opcional)**: Incluye un "switch" para aplicar el método de Macenko, estandarizando virtualmente las diferencias de tinción y colorimetría entre laboratorios para mejorar la consistencia en el análisis.

## 🛠️ Tecnologías y Requisitos

*   [Streamlit](https://streamlit.io/) - Framework de la UI interactiva.
*   [PyTorch](https://pytorch.org/) - Framework de Deep Learning para cargar los pesos y realizar la inferencia.
*   OpenCV y Pillow (PIL) - Modificación y estandarización visual de las capturas del microscopio.
*   NumPy / Pandas - Procesamiento numérico y métricas de probabilidad.

## 📥 Instrucciones de Instalación Local

1.  Clonar este repositorio en el sistema local:
    ```bash
    git clone https://github.com/EduardoTllo/breast-cancer-diagnosis-streamlit.git
    cd breast-cancer-diagnosis-streamlit
    ```

2.  *(Opcional pero recomendado)* Crear y activar un entorno virtual:
    ```bash
    python -m venv venv
    # En Windows: venv\Scripts\activate
    # En macOS/Linux: source venv/bin/activate
    ```

3.  Instalar todas las dependencias requeridas documentadas en `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  Iniciar el servidor web local de Streamlit:
    ```bash
    streamlit run app.py
    ```
5.  Se desplegará automáticamente una pestaña en tu navegador por defecto alojada en la dirección `http://localhost:8501`.

## ℹ️ Contexto del Dataset (BreaKHis)

Los modelos que impulsan esta aplicación están respaldados por el dataset **BreaKHis**, el cual está compuesto por:
*   **9,109 imágenes** microscópicas en total, etiquetadas y validadas.
*   Muestras pertenecientes a un conjunto de **82 pacientes**.
*   Capturas clínicas que provienen de 4 niveles de magnificación estandarizados: **40X, 100X, 200X y 400X**.

## ⚠️ Advertencia y Cláusula de Responsabilidad Médica

**Esta aplicación web ha sido creada de manera estrictamente académica a modo de prueba de concepto (PoC).** 
Los resultados, métricas y diagnósticos que el sistema pueda emitir están diseñados como una herramienta experimental y de apoyo. De ninguna manera suponen un dictamen clínico oficial ni **deben ser orientados como sustituto del diagnóstico calificado proporcionado por un médico especialista o patólogo profesional.**
