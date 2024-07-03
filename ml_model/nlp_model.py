import nltk
import spacy
from flask import Flask, request, jsonify

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    nltk_tokens = nltk.word_tokenize(text)
    processed_text = ' '.join(nltk_tokens)

    spacy_doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]

    return jsonify({
        'text': processed_text
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)