import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from BERTIntuitionModel import BERTIntuitionModel
from IntentDataset import IntentDataset
import random
import time
import matplotlib.pyplot as plt
import numpy as np

# Load vocabulary from file
def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# Create a custom tokenizer
def create_custom_tokenizer(vocab_file):
    vocab = load_vocab(vocab_file)
    return BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

# Load custom tokenizer
tokenizer = create_custom_tokenizer('fine_tuned_model/vocab.txt')

# Create a sample dataset (replace with your actual dataset)
sample_texts = [
    "What's the weather like today?",
    "Set an alarm for 7 AM",
    "Play some music",
    "What's on my calendar for tomorrow?",
    "Send a message to John",
    "What's the capital of France?",
    "How do I make pasta?",
    "Tell me a joke",
    "What's the time?",
    "Translate 'hello' to Spanish"
]

sample_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # Adjust according to your dataset

# Shuffle the samples
combined = list(zip(sample_texts, sample_labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# Initialize model
num_labels = len(set(labels))  # Dynamically set the number of labels
model = BERTIntuitionModel(num_labels)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create DataLoader
dataset = IntentDataset(texts, labels, tokenizer, max_len=128)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Fine-tuning setup
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(dataloader) * 3  # Assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Prepare directory for saving model
save_dir = 'fine_tuned_model'
os.makedirs(save_dir, exist_ok=True)

# Training loop
model.train()
epoch_losses = []
intuition_adjustments = []

for epoch in range(3):
    print(f"Epoch {epoch + 1}/3")
    epoch_start_time = time.time()
    epoch_loss = 0

    for step, batch in enumerate(dataloader):
        step_start_time = time.time()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs, intuition_output, layer_outputs = model(input_ids, attention_mask, step)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.intuition_nn.compare_and_adjust(layer_outputs, intuition_output)

        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        epoch_loss += loss.item()

        if step % 1 == 0:  # Print every step, adjust as needed
            print(f"Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}, Step Time: {step_time:.2f}s")

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    avg_epoch_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    intuition_adjustments.append(model.intuition_nn.intuition_coefficients.detach().cpu().numpy().copy())
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}, Epoch Time: {epoch_time:.2f}s")

# Save the final model
final_model_path = os.path.join(save_dir, "fine_tuned_intuition_model_final.pth")
torch.save(model.state_dict(), final_model_path)
tokenizer.save_pretrained(save_dir)

# Save vocab.txt in the fine_tuned_model directory
vocab_file_path = os.path.join(save_dir, 'vocab.txt')
with open(vocab_file_path, 'w', encoding='utf-8') as f:
    for word in tokenizer.get_vocab().keys():
        f.write(word + '\n')

print("Training completed. Final model and vocab.txt saved.")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, 4), epoch_losses, marker='o', label='Average Loss per Epoch')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'training_loss.png'))
plt.show()

# Plot intuition coefficient adjustments
intuition_adjustments = np.array(intuition_adjustments)
plt.figure(figsize=(10, 5))
for i in range(intuition_adjustments.shape[1]):
    plt.plot(range(1, 4), intuition_adjustments[:, i], label=f'Coeff {i+1}')
plt.title('Intuition Coefficient Adjustments Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'intuition_coefficients.png'))
plt.show()

# Print final metrics
print("Final Average Loss:", epoch_losses[-1])
print("Intuition Coefficient Adjustments:", intuition_adjustments[-1])
