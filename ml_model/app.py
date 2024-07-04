import nltk
import spacy
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from model import IntuitionNN

# Load NLTK and Spacy
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
intent_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define the IntuitionNN model
input_size = 768
layer_sizes = [128, 64, 32]
intuition_size = 10
model = IntuitionNN(input_size=input_size, layer_sizes=layer_sizes, intuition_size=intuition_size)
model.eval()

app = Flask(__name__)

# Manage conversation context
class Conversation:
    def __init__(self):
        self.state = {}

    def update_state(self, user_id, key, value):
        if user_id not in self.state:
            self.state[user_id] = {}
        self.state[user_id][key] = value

    def get_state(self, user_id, key):
        return self.state.get(user_id, {}).get(key, None)

conversations = Conversation()

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text')
    user_id = 'user'
    ai_id = 'ai'
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        embeddings = bert_model(**inputs).last_hidden_state[:, 0, :]

    # Recognize intent
    with torch.no_grad():
        intent_logits = intent_model(**inputs).logits
        intent_index = torch.argmax(intent_logits, dim=1).item()

    # Generate response using GPT-2
    prompt = text
    gpt_inputs = gpt_tokenizer.encode(prompt, return_tensors='pt')
    gpt_outputs = gpt_model.generate(gpt_inputs, max_length=50, num_return_sequences=1)
    response = gpt_tokenizer.decode(gpt_outputs[0], skip_special_tokens=True)

    # Update context
    conversations.update_state(user_id, 'last_intent', intent_index)
    conversations.update_state(ai_id, 'response', response)

    return jsonify({
        'text': response,
        'context': {
            'user_intent': conversations.get_state(user_id, 'last_intent'),
            'ai_response': conversations.get_state(ai_id, 'response')
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
