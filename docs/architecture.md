# System Architecture – Smart Chatbot App

This document outlines the technical architecture of the `smart-chatbot-app` project. It is intended for developers and researchers interested in understanding how the system is structured and how its components interact.

---

## High-Level Overview

```
[ Vue 3 Frontend ]
        │
        ▼
[ Node.js Middleware (Express + MongoDB) ]
        │
        ▼
[ Flask ML Backend (BERT + GPT-2 + DIAN) ]
```

---

## Frontend (Vue 3 SPA)

- Built using **Vue 3 + Vite**
- Single-page application for chat interface
- Communicates with backend through `/api/process-text`
- Shows conversation history from MongoDB (if enabled)
- Uses Tailwind CSS for responsive design

---

## Middleware Layer (Node.js + TypeScript + MongoDB)

- Express server that:
  - Receives user messages from frontend
  - Stores both user and AI messages in MongoDB
  - Forwards user text to Flask ML backend
  - Returns generated AI response to the client

### Data Flow:

1. `POST /api/process-text` from frontend
2. Saves message to MongoDB
3. Sends request to ML backend (`process.env.ML_API_URL`)
4. Receives and stores AI response
5. Returns response to frontend

---

## ML Backend (Flask + PyTorch)

### Technologies:

- Flask for serving REST API
- PyTorch for custom models
- HuggingFace Transformers (BERT, GPT-2)
- NLTK, spaCy for preprocessing

### Responsibilities:

- Classify intent using `BERTIntuitionModel`
- Generate response using GPT-2 based on prior context
- Learn from user inputs with online update (backpropagation)
- Update intuition coefficients to reflect internal uncertainty

---

## DIAN – Dynamic Input Activation Network

DIAN is an experimental neural mechanism embedded in the BERT-based classifier. It enables **runtime adaptation of model behavior** based on current inputs and past activations.

### Components:

- `IntuitionNN`: Multi-layer perceptron with optional skip connections
- `intuition_layer`: Learns expected internal activations
- `compare_and_adjust()`: Compares actual vs expected and adjusts weights/coefficients

### Behavior:

- During each prediction:
  - Layer activations are captured
  - Compared with stored "intuition"
  - A correction term is computed
  - Weights and `intuition_coefficients` are updated online

---

## Use Cases and Research Applications

- Experimental adaptive chatbot behavior
- Real-time model evolution visualization
- Online learning workflows
- Meta-learning studies
- Conversation analysis based on intuition metrics

---

## Data Storage

- **MongoDB** is used to persist all user and AI messages
- Messages include:
  - Text
  - Sender ("user" or "ai")
  - Timestamps
  - Session metadata (optional)

---

## Known Limitations

- Session state is not isolated per user
- Online updates are uncontrolled (no validation set)
- Intuition engine is heuristic and best used in sandbox settings

---

## Suggestions for Improvement

- Add user authentication and per-session memory
- Visualize intuition coefficient drift over time
- Implement input sanitation and validation
- Add CLI tools for data exploration and replay
