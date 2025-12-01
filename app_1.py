import streamlit as st
import numpy as np
import pydicom
import cv2
import tensorflow as tf
import os
import pandas as pd
from io import BytesIO
from tensorflow.keras.models import load_model as keras_load_model

# Configuramos las variables input
st.set_page_config(page_title="Predicción de Calcificaciones - TFM", layout="wide")
MODEL_PATH = "models/MobileNetV2/mobilenetv2_mammo_tf220.keras"
DATASET_BASE_PATH = "demo"

# Cargamos el modelo con el comando cache_resource para evitar recargar el modelo
# cada vez que se sube una imagen y así mejorar el rendimiento de la app
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el fichero de modelo en: {MODEL_PATH}")
        st.stop()

    try:
        # Cargamos el modelo completo tal y como se guardó en Colab
        model = keras_load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error cargando el modelo completo desde '{MODEL_PATH}': {e}")
        raise

    return model

model = load_model()

# Preprocesamiento para normalizar, aplicar CLAHE y redimensionar
def preprocess_dicom(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)
    image -= np.min(image)
    image /= (np.max(image) + 1e-8)
    image_uint8 = np.uint8(image * 255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(image_uint8)

    resized_img = cv2.resize(clahe_img, (224, 224), interpolation=cv2.INTER_AREA)
    rgb_img = np.stack([resized_img] * 3, axis=-1)
    rgb_img = rgb_img.astype(np.float32) / 255.0

    return image_uint8, clahe_img, np.expand_dims(rgb_img, axis=0)

# Barra lateral
st.sidebar.title("TFM: Detección de Calcificaciones")

# Branding
logo_path = os.path.join(DATASET_BASE_PATH, "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)
else:
    st.sidebar.warning(f"No se encontró el logo en {logo_path}")

st.sidebar.markdown(
    """
    <b>TFM:</b> Predicción de calcificaciones en mamografías<br>
    <b>Autor:</b> Elias Pallarès, Borja Nuñez y Martín Mazuera<br>
    <b>Tutor:</b> Alexandra Abós<br>
    <b>Universidad:</b> UPF-BSM<br>
    <b>Fecha:</b> Junio 2025
    """,
    unsafe_allow_html=True
)

# Selector de dataset e imagen de ejemplo
dataset_options = ["CBIS-DDSM", "CMMD"]
selected_dataset = st.sidebar.selectbox("Selecciona un dataset", dataset_options)
dataset_path = os.path.join(DATASET_BASE_PATH, selected_dataset)

available_images = []
if os.path.isdir(dataset_path):
    available_images = [f for f in os.listdir(dataset_path) if f.lower().endswith(".dcm")]
else:
    st.sidebar.warning(f"No existe la carpeta de dataset: {dataset_path}")

if available_images:
    selected_image = st.sidebar.selectbox("Selecciona una imagen de ejemplo", available_images)
    dicom_path = os.path.join(dataset_path, selected_image)
else:
    selected_image = None
    dicom_path = None
    st.sidebar.info("No hay imágenes de ejemplo disponibles en el repositorio. "
                    "Puedes subir tus propios DICOM por lote.")

# Opción para subir múltiples nuevas imágenes
dicom_batch_files = st.sidebar.file_uploader(
    "Sube imágenes DICOM por lote",
    type=["dcm"],
    accept_multiple_files=True
)

# Predicción individual
st.title("Predicción de Calcificaciones en Mamografías")

if dicom_path is not None and os.path.exists(dicom_path):
    try:
        original_img, clahe_img, input_img = preprocess_dicom(dicom_path)
        prediction = model.predict(input_img, verbose=0)[0][0]
        pred_label = "Maligno" if prediction > 0.5 else "Benigno"
        prob = round(float(prediction), 3)

        col1, col2 = st.columns(2)
        col1.image(
            clahe_img,
            caption=f"Imagen preprocesada (CLAHE) - {selected_image}",
            use_container_width=True
        )
        col2.markdown(
            f"### Predicción: `{pred_label}`\nProbabilidad de malignidad: `{prob}`"
        )

    except Exception as e:
        st.error(f"Error procesando la imagen de ejemplo: {e}")
else:
    st.info(
        "No se ha podido cargar una imagen de ejemplo. "
        "Verifica que existan archivos .dcm en las carpetas demo/CBIS-DDSM y demo/CMMD "
        "o sube tus propios DICOM desde la barra lateral."
    )

# Predicción por lotes
if dicom_batch_files:
    st.subheader("Predicciones por lote")
    results = []
    for file in dicom_batch_files:
        try:
            _, _, input_img = preprocess_dicom(file)
            prediction = model.predict(input_img, verbose=0)[0][0]
            label = "Maligno" if prediction > 0.5 else "Benigno"
            results.append({
                "Archivo": file.name,
                "Probabilidad": round(float(prediction), 3),
                "Clasificación": label
            })
        except Exception as e:
            results.append({
                "Archivo": file.name,
                "Probabilidad": "Error",
                "Clasificación": str(e)
            })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

    # Descargar en Excel
    towrite = BytesIO()
    with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Resultados')
    towrite.seek(0)
    st.download_button(
        "Descargar resultados Excel",
        data=totime,  # <-- aquí en tu código real asegúrate que se llama towrite, no time
        file_name="predicciones_batch.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Pie de página
st.markdown("---")
st.markdown("© 2025 Elias Pallarès Borja Nuñez, Martín Mazuera – TFM – UPF-BSM")