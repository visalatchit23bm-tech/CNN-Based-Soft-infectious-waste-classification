import numpy as np
from tensorflow import load_model
from PIL import Image

# Load the trained model
model = load_model("mobilenetv2_waste_model.keras")

# Load class names from text file
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

def predict_image(image):
    """
    Predict waste category from image.
    Returns: predicted_class (str), confidence (float), probabilities (np.array)
    """
    # Preprocess image
    image = image.resize((224, 224))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Model prediction
    prediction = model.predict(image, verbose=0)

    # Get highest probability index
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    predicted_class = class_names[predicted_index]

    # Probabilities for all classes
    probabilities = prediction[0]

    return predicted_class, confidence, probabilities
   

