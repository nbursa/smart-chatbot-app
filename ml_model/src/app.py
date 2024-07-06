import nltk
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel, set_seed
from models.IntuitionNN import IntuitionNN
from src.conversation import Conversation
from src.utils import log_conversation

# Load NLTK and Spacy
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
intent_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
set_seed(42)

# Define the IntuitionNN model
input_size = 768
layer_sizes = [128, 64, 32]
intuition_size = 10
model = IntuitionNN(input_size=input_size, layer_sizes=layer_sizes, intuition_size=intuition_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

app = Flask(__name__)

conversations = Conversation()

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text')
    user_id = 'user'
    ai_id = 'ai'
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Recognize intent
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        embeddings = bert_model(**inputs).last_hidden_state[:, 0, :]
        intent_logits = intent_model(**inputs).logits
        intent_index = torch.argmax(intent_logits, dim=1).item()

    # Generate response using GPT-2 with context
    context = conversations.get_context(user_id)
    prompt = f"{context}\nUser: {text}\nAI:"
    gpt_inputs = gpt_tokenizer.encode(prompt, return_tensors='pt')
    gpt_outputs = gpt_model.generate(gpt_inputs, max_length=100, num_return_sequences=1, pad_token_id=gpt_tokenizer.eos_token_id, top_k=50, top_p=0.95)
    response = gpt_tokenizer.decode(gpt_outputs[0], skip_special_tokens=True)

    # Post-process the response to make it more coherent
    response = response.replace(prompt, "").strip()
    response = response.split('\n')[0]  # Take only the first line for simplicity

    # Update context
    conversations.update_state(user_id, 'User', text)
    conversations.update_state(user_id, 'AI', response)

    # Train IntuitionNN
    model.train_step(embeddings, torch.tensor([intent_index]), optimizer, criterion)

    # Log conversation
    log_conversation(user_text=text, ai_text=response, intent=intent_index)

    return jsonify({
        'text': response
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
