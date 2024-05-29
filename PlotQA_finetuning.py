import json
import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset as HFDataset

json_path = './deplot_format_label.json' # annotation 이름 수정해줘야 함
with open(json_path, 'r') as f:
    data = json.load(f)

data_list = [{"image": os.path.join('./chart_image/', f"{idx}.png"), "text": text} for idx, text in data.items()]
df = pd.DataFrame(data_list[:300])

dataset = HFDataset.from_pandas(df)
print(dataset[0]['image'])

processor = Pix2StructProcessor.from_pretrained('google/deplot')

def load_image(image_path):
    return Image.open(image_path)

# def tokenize_example(example):
#     image = load_image(example["image"])
#     return processor(images=image, text=example["text"], return_tensors="pt")

# tokenized_dataset = dataset.map(tokenize_example, batched=False)
# train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
# train_dataset = train_test_split['train']
# eval_dataset = train_test_split['test']

MAX_PATCHES = 1024

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = load_image(item["image"])  # Load the image here
        encoding = self.processor(images=image, text=item["text"], return_tensors="pt", add_special_tokens=True)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["text"] for item in batch]

    text_inputs = processor.tokenizer(text=texts, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True, max_length=20)

    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

def main():
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

    EPOCHS = 50
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = "./training_log.txt"

    model.train()
    for epoch in range(EPOCHS):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            outputs = model(flattened_patches=flattened_patches,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss

            print("Loss:", loss.item())

            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item()}\n")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.bin")
        torch.save(model.state_dict(), model_save_path)

        if (epoch + 1) % 10 == 0:
            model.eval()
            predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)
            print("Predictions:", processor.batch_decode(predictions, skip_special_tokens=True))
            model.train()

if __name__ == "__main__":
    main()