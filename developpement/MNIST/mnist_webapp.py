import gradio as gr
from PIL import Image
import requests
import io
import numpy as np


def recognize_digit(image):
    # Extract the composite image from the sketchpad dictionary
    if isinstance(image, dict) and 'composite' in image:
        image = image['composite']
    
    # Convert to PIL Image
    if hasattr(image, 'astype'):
        # If it's a numpy array
        image = Image.fromarray(image.astype('uint8'))
    else:
        # If it's already a PIL Image
        image = image
    
        # Convert to grayscale to ensure single channel
    if image.mode is None:
        pass
    elif image.mode != 'L':
        image = image.convert('L')

    img_binary = io.BytesIO() #convertion fichier binaire 
    image.save(img_binary, format='PNG')
    response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
    return response.json()["prediction"]

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);