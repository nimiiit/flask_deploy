from flask import Flask, request, jsonify
from app.utils import transform_image, get_prediction

app=Flask(__name__)

ALLOWED_EXT={'png', 'jpg', 'jpeg'}
Classes=['female','male']
def allowed_file(filename):
    print(filename.rsplit('.',1)[1].lower())
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file= request.files.get('file')
        if file is None or file.filename=="":
            return jsonify({"error": 'no file' })
        if not allowed_file(file.filename):
            return jsonify({"error": 'file format not supported'})

        try:
            img_bytes=file.read()
            tensor=transform_image(img_bytes)
            prediction =get_prediction(tensor)
            return jsonify({"prediction": prediction.item(), "class": Classes[int(prediction.item())]})
        except:
            return jsonify({"error": 'error during prediction'})    
  