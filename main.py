from flask import Flask, request, render_template, send_from_directory
import numpy as np
import os
import pickle
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'images'
CACHE_FILE = 'features_cache.pkl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load or create feature cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        features_cache = pickle.load(f)
else:
    features_cache = {}

# Load the VGG16 model for feature extraction
feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match model's expected input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_features(image_path):
    img = preprocess_image(image_path)
    features = feature_extractor.predict(img)
    return features.flatten()

def index_images(image_folder):
    features = {}
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(image_folder, filename)
            if filename not in features_cache:
                features_cache[filename] = extract_features(path)
            features[filename] = features_cache[filename]
    
    # Save the updated cache
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(features_cache, f)
        
    return features

def search_image(query_image, indexed_features, top_n=10):
    query_feature = extract_features(query_image)
    
    # Use NearestNeighbors for faster searching
    nbrs = NearestNeighbors(n_neighbors=top_n, algorithm='auto').fit(list(indexed_features.values()))
    distances, indices = nbrs.kneighbors([query_feature])
    
    results = [list(indexed_features.keys())[i] for i in indices[0]]
    return results

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Index images and search for similar
    indexed_features = index_images(IMAGE_FOLDER)
    results = search_image(file_path, indexed_features, top_n=10)

    return render_template('result.html', results=results)

@app.route('/images/<filename>')
def image_file(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
