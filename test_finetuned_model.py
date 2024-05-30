import json
import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset as HFDataset

processor = Pix2StructProcessor.from_pretrained('google/deplot')

def load_trained_model(checkpoint_path, processor):
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def predict(image_path, model, processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    predictions = model.generate(**inputs)
    decoded_predictions = processor.batch_decode(predictions, skip_special_tokens=True)
    return decoded_predictions

# Load the trained model from a specific checkpoint
checkpoint_path = "./checkpoints/model_epoch_39.bin"  # Specify the checkpoint you want to load
trained_model = load_trained_model(checkpoint_path, processor)

# Perform inference on a new image
test_image_path = "./path_to_your_test_image.png"  # Specify the path to the test image
prediction = predict(test_image_path, trained_model, processor)
print("Prediction for the test image:", prediction)