import nltk
import spacy
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Setup spaCy model
nlp = spacy.load("en_core_web_sm")

# Endpoint for processing text
@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenization using NLTK
    nltk_tokens = nltk.word_tokenize(text)

    # Named Entity Recognition using spaCy
    spacy_doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]

    return jsonify({
        'nltk_tokens': nltk_tokens,
        'spacy_entities': entities
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)