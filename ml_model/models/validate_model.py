import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from BERTIntuitionModel import BERTIntuitionModel
from IntentDataset import IntentDataset

# Load the trained model
model_path = 'fine_tuned_model/fine_tuned_intuition_model_final.pth'
tokenizer = BertTokenizer.from_pretrained('fine_tuned_model')
num_labels = 5  # Update this to the correct number of labels in your dataset

# Initialize the model with the correct number of labels
model = BERTIntuitionModel(num_labels)
model.load_state_dict(torch.load(model_path))
model.eval()

# Prepare validation dataset (replace with your actual validation texts and labels)
validation_texts = [
    "What is the weather tomorrow?",
    "Wake me up at 6 AM",
    "Play jazz music",
    "Show my schedule for the week",
    "Text Jane",
    "What is the capital of Italy?",
    "Recipe for pancakes",
    "Tell a funny story",
    "Current time?",
    "How do you say 'goodbye' in French?"
]

validation_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # Adjust according to your dataset

# Create DataLoader
validation_dataset = IntentDataset(validation_texts, validation_labels, tokenizer, max_len=128)
validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=False)

# Evaluate model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in validation_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs, _, _ = model(input_ids, attention_mask, 0)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_labels), yticklabels=range(num_labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
