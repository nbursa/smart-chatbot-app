import nltk
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel, set_seed
from models.BERTIntuitionModel import BERTIntuitionModel

from src.conversation import Conversation
from src.utils import log_conversation

# Load NLTK and Spacy
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
set_seed(42)

# Define the BERTIntuitionModel
num_labels = 5
model = BERTIntuitionModel(num_labels)

# Load the fine-tuned model if available
try:
    model.load_state_dict(torch.load("fine_tuned_intuition_model_final.pth"))
    print("Loaded fine-tuned model")
except FileNotFoundError:
    print("No fine-tuned model found, using initial model")

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

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

    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Get intent using BERTIntuitionModel
    model.eval()
    with torch.no_grad():
        outputs, _, _ = model(inputs['input_ids'], inputs['attention_mask'], 0)
        intent_index = torch.argmax(outputs, dim=1).item()

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

    # Train BERTIntuitionModel
    model.train()
    optimizer.zero_grad()
    outputs, intuition_output, layer_outputs = model(inputs['input_ids'], inputs['attention_mask'], 0)
    loss = F.cross_entropy(outputs, torch.tensor([intent_index]))
    loss.backward()
    optimizer.step()
    model.intuition_nn.compare_and_adjust(layer_outputs, intuition_output)

    # Log conversation
    log_conversation(user_text=text, ai_text=response, intent=intent_index)

    return jsonify({
        'text': response
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
