import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("Model/mobilenetv2_waste_model.h5")

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

def predict_image(image):

    image = image.resize((224,224))

    image = image.convert("RGB")

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction)*100)

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence

