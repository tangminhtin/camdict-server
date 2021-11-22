# Required Imports
import os
import tensorflow as tf
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from data import classes_names

# Initialize Flask App
app = Flask(__name__)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
vocabularies_ref = db.collection('vocabularies')

# Upload image
UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
saved_model_path = "model/fruits_minhtin.h5"
model = tf.keras.models.load_model(saved_model_path)


# Load image
def load_image(filename, img_shape=224, scale=True):
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img)  # Decode it into a tensor
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img/255.
    else:
        return img

@app.route('/', methods=['GET'])
def home(): 
    return '<h1><center>Welcome to Camdict!</center></h1>'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'there is no file in form!'
    file = request.files['file']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Load and predict
    image = load_image(image_path, scale=False)
    pred_prob = model.predict(tf.expand_dims(image, axis=0))
    pred_class = classes_names[pred_prob.argmax()]
    # Plot the image with appropriate annotations
    print(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")

    if (pred_prob.max() >= 0.85):
        # Get id and find vocabulary
        vocabularies_id = pred_class.get('id')
        vocabulary = vocabularies_ref.document(vocabularies_id).get()
        return jsonify(vocabulary.to_dict()), 200

    else:
        return jsonify({"msg": 'error'}), 500


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
   from waitress import serve
   serve(app, host="0.0.0.0", port=port)
