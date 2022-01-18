from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np

from feature_extractor import FeatureExtractor

app = Flask(__name__)

base_dir = './static/img'
feature_dir = './static/feature'
model_dir = './static/model'

# Read image features
fe = FeatureExtractor(load_model(model_dir + '/ModelCNN25.h5', compile=False))

img_paths = list()

for img_path in sorted(Path(base_dir).glob("*.jpeg")):
    img_paths.append(img_path)

features = np.load(feature_dir + '/extracted_feature.npy')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['query_img']
            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        except:
            filename = request.form.get('query_img')
            img = Image.open(filename)
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + filename.split('/')[-1]

        # Save query image
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:12]  # Top 9 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
