import streamlit as st
import os
import cv2
import pytesseract
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Função para verificar as extensões permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Carregar o modelo
model = YOLO('best_license_plate_model.pt')

def process_image(image):
    # Converter a imagem do PIL para formato que o OpenCV pode manipular
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Realizar a predição na imagem
    results = model.predict(image_cv, device='cpu')

    # Converter a imagem de volta para RGB para exibição no Streamlit
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    detected_texts = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_rgb, f'{confidence*100:.2f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            roi = image_rgb[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, config='--psm 6')
            detected_texts.append((text, confidence))

    return image_rgb, detected_texts

# Interface do Streamlit
st.title("Detecção de Placas de Carro")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg", "gif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Imagem carregada', use_column_width=True)
    st.write("Processando...")

    processed_image, detected_texts = process_image(image)

    st.image(processed_image, caption='Imagem Processada', use_column_width=True)

    st.write("Textos Detectados:")
    for text, confidence in detected_texts:
        st.write(f"**Texto:** {text}")
        st.write(f"**Confiança:** {confidence*100:.2f}%")
