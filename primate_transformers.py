import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import json
import numpy as np
import re
from torch.utils.data import DataLoader
from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report


# Load and preprocess data
class MentalHealthDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item['labels'] = torch.tensor(self.labels[idx])
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)
    


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        #self.bert = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs[1])
        logits = self.sigmoid(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

        return ((loss,) + (logits,)) if loss is not None else (logits,)
    
# Check if a GPU is available and if PyTorch is using it
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name())
else:
    print("Using CPU")

with open("primate_dataset.json", "r") as file:
    data = json.load(file)
texts = [item['post_text'] for item in data]

# Convert annotations to list of symptoms with "yes" label
labels = [[symptom[0] for symptom in post['annotations'] if symptom[1] == 'yes'] for post in data]

# Convert labels to binary
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

symptom_names = mlb.classes_
#xprint(f'\n SYMPTOM NAMES: {symptom_names}')

# Tokenize texts
#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base")
encodings = tokenizer(texts, truncation=True, padding=True)



from sklearn.model_selection import train_test_split

# Get the number of samples
n_samples = len(encodings['input_ids'])

# Create a list of indices and split it
indices = list(range(n_samples))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Use the indices to split the encodings and labels
train_encodings = {key: [val[i] for i in train_indices] for key, val in encodings.items()}
test_encodings = {key: [val[i] for i in test_indices] for key, val in encodings.items()}
train_labels = [labels[i] for i in train_indices]
test_labels = [labels[i] for i in test_indices]

# Create training and test datasets
train_dataset = MentalHealthDataset(train_encodings, train_labels)
test_dataset = MentalHealthDataset(test_encodings, test_labels)

# Define model
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))
# Define model
model = BertForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))
# Enable gradient checkpointing
model.config.gradient_checkpointing = True


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=4,  # Reduce batch size
    gradient_accumulation_steps=4,  # Add gradient accumulation
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="none",  # Disable all integrations
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
# Get the first batch
batch = next(iter(DataLoader(train_dataset, batch_size=8)))

# Print the shapes of the input and label tensors
print("Input shape:", batch['input_ids'].shape)
print("Label shape:", batch['labels'].shape)

# Train model
trainer.train()

eval_result = trainer.evaluate()

# Get predictions
predictions = trainer.predict(test_dataset)

# Convert logits to 0s and 1s
all_preds = np.where(predictions.predictions > 0.7, 1, 0)

# Get true labels
all_labels = predictions.label_ids

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
roc_auc = roc_auc_score(all_labels, all_preds, average='macro')

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
print("\n#################### Classification Report ####################\n")
# Generate classification report
report = classification_report(all_labels, all_preds, target_names=symptom_names)

# Print report
print(report)

mcc_scores = [matthews_corrcoef(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]

# # Print MCC scores
# for i, mcc in enumerate(mcc_scores):
#     print(f"MCC for symptom {i}: {mcc}")

# Print MCC scores
for i, mcc in enumerate(mcc_scores):
    print(f"MCC for symptom {symptom_names[i]}: {mcc}")