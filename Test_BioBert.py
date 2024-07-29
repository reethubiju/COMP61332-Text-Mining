import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import re, json
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
# Load the test data from a JSONL file
test_data_path = './chemprot_test.jsonl'
test_data = []
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('./biobert_tokenizer')  # Update this path

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('./biobert_finetuned')  # Update this path


with open(test_data_path, 'r') as file:
    for line in file:
        test_data.append(json.loads(line))

# Convert to DataFrame
test_df = pd.DataFrame(test_data)

# Apply the same preprocessing to the 'text' column of the test data
test_df['preprocessed_text'] = test_df['text'].apply(preprocess_text)





# Assuming the tokenizer is already initialized and the preprocess_text function is defined

# Tokenize the test data
test_encodings = tokenizer(test_df['preprocessed_text'].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

# Convert to a Dataset
class TestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings.input_ids[idx],
            'attention_mask': self.encodings.attention_mask[idx]
        }

test_dataset = TestDataset(test_encodings)

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)






import torch

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()  # Set the model to evaluation mode

predictions = []
# No need for true labels if we're only predicting
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
print(predictions)

# Specify the file name
file_name = "output_biobert.txt"
# Convert the list/array elements to strings and join them with commas
output_string = ','.join(map(str, predictions))
# Open the file in write mode
with open(file_name, 'w') as f:
    # Write the output string to the file
    f.write(output_string)
print("List/array has been written to", file_name)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Ensure true_labels and predictions are aligned and have the same length
# Convert them to the correct format if necessary (e.g., NumPy arrays or Python lists)
label_encoder = preprocessing.LabelEncoder()
test_df['encoded_labels'] = label_encoder.fit_transform(test_df['label'].values)
# Calculate metrics
accuracy = accuracy_score(test_df['encoded_labels'], predictions)
precision = precision_score(test_df['encoded_labels'], predictions, average='weighted')
recall = recall_score(test_df['encoded_labels'], predictions, average='weighted')
f1 = f1_score(test_df['encoded_labels'], predictions, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

