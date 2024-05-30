import torch
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, TrainingArguments, Trainer

### path ###
checkpoint_path = "weights/reform2_epoch_34.bin"  # Specify the checkpoint you want to load
test_image_path = "./png_and_annotation_100/0.png"  # Specify the path to the test image
###



def load_trained_model(checkpoint_path):
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def predict(image_path, model):
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    image = Image.open(image_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to(device)
    predictions = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(predictions[0], skip_special_tokens=True)

def print_prediction(prediction):
    for line in prediction.split('0x0A'):
        print(line)

trained_model = load_trained_model(checkpoint_path)
prediction = predict(test_image_path, trained_model)
print("Prediction for the test image:")
print('\n---------------\n')
print_prediction(prediction)
print('\n---------------\n')
