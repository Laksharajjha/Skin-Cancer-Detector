from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load trained model
model = load_model('/Users/laksharajjha/Desktop/model/skin_cancer_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the file to the static folder
            img_path = 'static/' + file.filename
            file.save(img_path)

            # Preprocess the image
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(img_array)

            # Debug: Print the raw model output
            print("Raw model output:", prediction)

            # Binary classification logic: If the output is > 0.5, classify as 'Positive'
            result = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

            # Render result in the template
            return render_template('index.html', result=result, image_path=img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
