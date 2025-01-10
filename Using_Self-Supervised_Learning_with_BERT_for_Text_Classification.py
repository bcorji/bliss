import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Sample data
data = {
    'text': ["I love programming.", "Python is great for data science.", "I dislike bugs in my code.", "Machine learning is fascinating.", "Debugging is tedious."],
    'label': [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the data
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

tokenized_texts = [tokenize_function({'text': text}) for text in df['text']]

# Convert to tensor format
input_ids = torch.tensor([t['input_ids'] for t in tokenized_texts])
attention_masks = torch.tensor([t['attention_mask'] for t in tokenized_texts])
labels = torch.tensor(df['label'].values)

# Split the data
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, test_masks = train_test_split(attention_masks, test_size=0.2, random_state=42)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_inputs, train_masks, train_labels)
test_dataset = Dataset(test_inputs, test_masks, test_labels)

# Load a pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer.evaluate()