pip install flask
pip install opencv-python-headless
pip install numpy

from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

def compare_faces(image1, image2):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale images
    faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Compare the number of detected faces
    if len(faces1) > 0 and len(faces2) > 0:
        return len(faces1) == len(faces2)
    else:
        return False

@app.route('/compare_faces', methods=['POST'])
def perform_face_comparison():
    try:
        image1 = request.files.get('image1')
        image2 = request.files.get('image2')

        # Read the images using OpenCV
        nparr1 = np.fromstring(image1.read(), np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        nparr2 = np.fromstring(image2.read(), np.uint8)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        # Perform face comparison
        is_matched = compare_faces(img1, img2)

        response = {'is_matched': is_matched}
        return jsonify(response)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
