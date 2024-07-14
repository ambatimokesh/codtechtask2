# codtechtask2
TEXT--TO--SPEECH CONVERSTION.

import torch
from transformers import DalleBartProcessor, DalleBartForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Load the processor and model
processor = DalleBartProcessor.from_pretrained("flax-community/dalle-mini")
model = DalleBartForConditionalGeneration.from_pretrained("flax-community/dalle-mini")

def generate_image_from_text(description):
    # Encode the input text
    inputs = processor(description, return_tensors="pt")

    # Generate the image
    outputs = model.generate(**inputs)

    # Decode the output to images
    images = processor.batch_decode(outputs, skip_special_tokens=True)

    return images

def display_image(image):
    img = Image.open(BytesIO(requests.get(image).content))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def main():
    description = input("Enter a textual description: ")
    images = generate_image_from_text(description)
    
    for i, img_url in enumerate(images):
        print(f"Image {i+1}:")
        display_image(img_url)

if __name__ == "__main__":
    main()


    FEATURES:
    * It can generate audio from text.
    * You can also save the audio with extension mp3.
    * easily generate the audio has a ouput.


    INSTALLATION:
    * pip install torch, to run then program with out any bugs.
    
