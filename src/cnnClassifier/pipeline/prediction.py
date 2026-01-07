import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load and preprocess image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        preds = model.predict(img_array)

        
        normal_prob = float(preds[0][0])
        tumor_prob = float(preds[0][1])


        tumor_threshold = 0.3 
        if tumor_prob >= tumor_threshold:
            predicted_label = "Tumor"
        else:
            predicted_label = "Normal"


        response = {
            "label": predicted_label,
            "normal_probability": normal_prob,
            "tumor_probability": tumor_prob,
            "threshold_used": tumor_threshold,
            "summary": f"Predicted class: {predicted_label}",
            "details": {
                "normal_percent": f"{normal_prob * 100:.2f}%",
                "tumor_percent": f"{tumor_prob * 100:.2f}%",
            },
        }

        return [response]
