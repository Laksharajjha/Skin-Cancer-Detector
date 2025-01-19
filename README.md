# Deep Learning Model for Skin Cancer Detection (Supervised Learning)

This project implements a Convolutional Neural Network (CNN)-based deep learning model using TensorFlow to detect skin cancer from medical images. The model is designed for high accuracy and reliability, aiming to assist in the early detection of skin cancer.

## Features
- **Skin Cancer Detection**: Detects potential skin cancer lesions from medical images.
- **Image Preprocessing**: Preprocesses images to optimize model training and performance.
- **Model Architecture**: Uses a fine-tuned CNN model for improved accuracy and reliability.
- **Flask Web Application**: A user-friendly web interface to input images and display predictions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Laksharajjha/SkinCancerDetection.git
2. Navigate to the project directory:
   cd SkinCancerDetection
3. Install the required dependencies:
   pip install -r requirements.txt
4. Run the Flask web application:
   python app.py
5. Access the app in your browser at http://127.0.0.1:5000/.
Model Details

Model Type: CNN (Convolutional Neural Network)
Framework: TensorFlow
Data Preprocessing: Standardizes images for optimal model performance
How It Works

Input: Users can upload a skin lesion image through the web application.
Prediction: The model processes the image and predicts whether the lesion is benign or malignant.
Output: The application displays the prediction along with the associated probability.
Contributing

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes.
Commit your changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-branch).
Open a pull request.
License

This project is licensed under the MIT License - see the LICENSE file for details.
