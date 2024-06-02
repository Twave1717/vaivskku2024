
import os
import json
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from transformers.optimization import Adafactor
from datasets import Dataset as HFDataset
from accelerate import Accelerator
from torch.cuda.amp import GradScaler, autocast
import gc

# Initialize Accelerator and GradScaler
accelerator = Accelerator(gradient_accumulation_steps=4)
scaler = GradScaler()

# Load data from JSON file
json_path = '/content/drive/MyDrive/train_annotation.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Prepare data list and dataframe
data_list = [{"image": os.path.join('/content/train_image/png', f"{idx}.png"), "text": text} for idx, text in data.items()]
df = pd.DataFrame(data_list[:300])

# Convert to Hugging Face Dataset
dataset = HFDataset.from_pandas(df)

# Load processor
processor = Pix2StructProcessor.from_pretrained('google/deplot')

# Function to load image
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# Custom Dataset class
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = load_image(item["image"])  # Load the image here
        encoding = self.processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt", add_special_tokens=True, max_patches=2048)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

# Collator function for DataLoader
def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["text"] for item in batch]

    text_inputs = processor.tokenizer(text=texts, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True, max_length=512)

    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch


# Main function for training
def main():
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=collator)

    EPOCHS = 50
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=0.0001, weight_decay=1e-05)
    device = accelerator.device
    model.to(device)

    checkpoint_dir = "/content/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = "/content/training_log.txt"

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model.train()

    for epoch in range(EPOCHS):
        print("Epoch:", epoch)
        optimizer.zero_grad()
        for idx, batch in enumerate(train_dataloader):
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            with autocast():
                outputs = model(flattened_patches=flattened_patches,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss / accelerator.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (idx + 1) % accelerator.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()

            print("Loss:", loss.item())
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item()}\n")


        model_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.bin")
        torch.save(model.state_dict(), model_save_path)
        model.train()

        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
              predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=1024)
              print("Predictions:", processor.batch_decode(predictions, skip_special_tokens=True))
            model.train()

if __name__ == "__main__":
    main()

