import streamlit as st
import torch
import os
import numpy as np
import pandas as pd
from utils import load_model, preprocess_image, GradCAM, overlay_cam

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de Cáncer de Mama",
    page_icon="🎗️",
    layout="wide"
)

# --- SIDEBAR & CONFIG ---
st.sidebar.title("Configuración")
st.sidebar.info("Esta aplicación utiliza una red neuronal ResNet18 para clasificar imágenes histopatológicas.")

# Opción de Macenko
use_macenko = st.sidebar.checkbox("Activar Normalización de Macenko", value=False, help="Aplica normalización de color para estandarizar la tinción de las imágenes.")

st.sidebar.divider()
st.sidebar.subheader("ℹ️ Sobre el Modelo")
st.sidebar.markdown("""
El modelo fue entrenado con el dataset **BreaKHis**, que contiene:
- **9,109** imágenes microscópicas de tejido tumoral mamario.
- **82** pacientes.
- Magnificaciones: **40X, 100X, 200X, 400X**.
- **2,480** muestras benignas y **5,429** malignas.

⚠️ **Advertencia**: Esta herramienta es un sistema de apoyo al diagnóstico y **no sustituye** la evaluación de un patólogo profesional.
""")

# Cargar Modelo
@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Asumimos que el modelo está en la misma carpeta o ruta relativa conocida
    model_path = os.path.join(os.path.dirname(__file__), "resnet18_focal_best.pth")
    return load_model(model_path, device), device

try:
    model, device = get_model()
    st.sidebar.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

# --- MAIN UI ---
st.title("🎗️ Sistema de Diagnóstico Asistido por IA")
st.markdown("""
Sube una o varias imágenes histopatológicas para obtener una predicción.
El sistema analizará si el tejido presenta características **Benignas** o **Malignas**.
""")

st.warning("📢 **Recomendación**: Si sube múltiples imágenes para aprovechar el **diagnóstico por votación**, asegúrese de que todas pertenezcan al **mismo paciente**.")

# Inicializar key en session state si no existe
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def clear_images():
    st.session_state["uploader_key"] += 1

uploaded_files = st.file_uploader("Subir imágenes (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key=f"uploader_{st.session_state['uploader_key']}")

if uploaded_files:
    st.button("🗑️ Borrar imágenes", on_click=clear_images)

    st.divider()
    st.subheader("Resultados del Análisis")
    
    # Contenedores para resultados
    results = []
    
    # Barra de progreso si son muchas imágenes
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        # Preprocesar
        img_tensor, img_pil = preprocess_image(file, use_macenko=use_macenko)
        img_tensor = img_tensor.to(device)
        
        # Inferencia
        grad_cam = GradCAM(model, model.layer4) # ResNet layer4 es la última conv
        mask, output = grad_cam(img_tensor)
        
        # Probabilidades
        probs = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        label = "Maligno" if pred_idx == 1 else "Benigno"
        color = "red" if pred_idx == 1 else "green"
        
        # Guardar resultado
        results.append({
            "filename": file.name,
            "prediction": label,
            "confidence": confidence,
            "probs": probs,
            "img_pil": img_pil,
            "cam_mask": mask
        })
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    
    # --- NIVEL PACIENTE (VOTACIÓN) ---
    if len(uploaded_files) > 1:
        st.info(f"Se analizaron {len(uploaded_files)} imágenes. Calculando diagnóstico global del paciente...")
        
        malignant_count = sum(1 for r in results if r["prediction"] == "Maligno")
        benign_count = len(results) - malignant_count
        
        # Promedio de probabilidades
        avg_probs = np.mean([r["probs"] for r in results], axis=0)
        global_pred_idx = np.argmax(avg_probs)
        global_label = "Maligno" if global_pred_idx == 1 else "Benigno"
        global_conf = avg_probs[global_pred_idx]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Votos Benignos", f"{benign_count}", delta=None)
        col2.metric("Votos Malignos", f"{malignant_count}", delta_color="inverse")
        col3.metric("Diagnóstico Global (Promedio Probs)", global_label, f"{global_conf:.2%}")
        
        if global_label == "Maligno":
            st.error("⚠️ El análisis sugiere un diagnóstico **MALIGNO** basado en las imágenes proporcionadas.")
        else:
            st.success("✅ El análisis sugiere un diagnóstico **BENIGNO** basado en las imágenes proporcionadas.")
            
        st.divider()

    # --- MOSTRAR DETALLES INDIVIDUALES ---
    st.subheader("Detalle por Imagen")
    
    # Mostrar en grid de 2 columnas
    cols = st.columns(2)
    
    for i, res in enumerate(results):
        with cols[i % 2]:
            st.markdown(f"**{res['filename']}**")
            
            # Crear overlay
            cam_img = overlay_cam(res['img_pil'], res['cam_mask'], alpha=0.4)
            
            # Mostrar lado a lado (Original vs CAM)
            # st.image([res['img_pil'], cam_img], caption=["Original", "Grad-CAM"], width=300) 
            # Mejor mostrar solo la CAM overlay para ahorrar espacio, o un toggle
            
            st.image(cam_img, caption=f"Predicción: {res['prediction']} ({res['confidence']:.2%})", use_container_width=True)
            
            if res['prediction'] == 'Maligno':
                st.markdown(f":red[**Maligno**] ({res['confidence']:.2%})")
            else:
                st.markdown(f":green[**Benigno**] ({res['confidence']:.2%})")
            
            with st.expander("Ver original y mapa de calor"):
                c1, c2 = st.columns(2)
                c1.image(res['img_pil'], caption="Original", use_container_width=True)
                c2.image(res['cam_mask'], caption="Mapa de Calor", clamp=True, use_container_width=True)

else:
    st.info("Por favor, sube imágenes para comenzar el análisis.")
