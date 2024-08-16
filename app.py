from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import pytesseract
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Configuração básica do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Função para verificar as extensões permitidas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

model = YOLO('best_license_plate_model.pt')

# Rota principal - Upload da imagem
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('process_image', filename=filename))
    return render_template('upload.html')

# Rota para processar a imagem e mostrar o resultado
@app.route('/process/<filename>')
def process_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    results = model.predict(image_path, device='cpu')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_texts = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            roi = image[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, config='--psm 6')
            detected_texts.append((text, confidence))

    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
    plt.imsave(processed_image_path, image)

    return render_template('process.html', filename='processed_' + filename, texts=detected_texts)

# Servir a imagem processada
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
