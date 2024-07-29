import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import re
from sklearn import preprocessing
from tqdm.auto import tqdm
def preprocess_text(text):
    patterns_replacements = {
        r"<<[^>>]+>>": "[ENTITY]", 
        r"\[\[[^\]]+\]\]": "[ENTITY2]"
    }
    for pattern, replacement in patterns_replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
# Load the dataset
df = pd.read_csv('./Merged.csv')

# Preprocess the dataset
df['preprocessed_text'] = df['text'].apply(preprocess_text)

# Initialize the BioBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Prepare labels (assuming you have a 'label' column with integer-encoded labels)
# This part may need to be adjusted based on how your labels are structured
label_encoder = preprocessing.LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['label'].values)

# Initialize dataset
dataset = CustomDataset(df['preprocessed_text'].tolist(), df['encoded_labels'].tolist(), tokenizer)

# DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)
# Model initialization
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=len(df['encoded_labels'].unique()))
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
model.train()
for epoch in range(3):  # Adjust number of epochs as needed
    loop = tqdm(loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# Save the model and tokenizer
model_save_path = "./biobert_finetuned"
tokenizer_save_path = "./biobert_tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print("Model and tokenizer have been saved.")
