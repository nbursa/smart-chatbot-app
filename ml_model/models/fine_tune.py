import os
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
from models.IntuitionNN import IntuitionNN

# Initialize models and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
input_size = 768
layer_sizes = [128, 64, 32]
intuition_size = 10
model = IntuitionNN(input_size=input_size, layer_sizes=layer_sizes, intuition_size=intuition_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def fine_tune_model(log_dir='data/conversation_logs'):
    for log_file in os.listdir(log_dir):
        with open(os.path.join(log_dir, log_file), 'r') as f:
            log_data = json.load(f)
            user_text = log_data['user']
            intent = log_data['intuition']['intent']
            inputs = tokenizer(user_text, return_tensors='pt')
            with torch.no_grad():
                embeddings = bert_model(**inputs).last_hidden_state[:, 0, :]
            loss = model.train_step(embeddings, torch.tensor([intent]), optimizer, criterion)
            print(f"Trained on {log_file}, Loss: {loss}")

if __name__ == "__main__":
    fine_tune_model()
