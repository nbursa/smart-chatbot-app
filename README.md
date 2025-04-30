# Smart Chatbot App

[![Status](https://img.shields.io/badge/status-experimental-yellow)](https://github.com/nbursa/smart-chatbot-app)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Vue 3](https://img.shields.io/badge/vue-3.x-42b883?logo=vue.js)](https://vuejs.org/)
[![Tailwind CSS](https://img.shields.io/badge/tailwindcss-3.x-38b2ac?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/backend-flask-red)](api/)
[![Node.js](https://img.shields.io/badge/node.js-18+-lightgreen)](https://nodejs.org/)
[![MongoDB](https://img.shields.io/badge/database-mongodb-4ea94b?logo=mongodb&logoColor=white)](https://www.mongodb.com/)

[Architecture](docs/architecture.md) • [AI Design](docs/ai-design.md) • [Contributing](CONTRIBUTING.md) • [License](LICENSE)

Published on Ready Tensor:  
[Smart Chatbot App – Adaptive NLP Agent with Intuition Modeling (DIAN)](https://app.readytensor.ai/publications/smart-chatbot-app-adaptive-nlp-agent-with-intuition-modeling-dian-wLYPCu6oX4Ce)

A full-stack web application featuring a machine learning-powered chatbot with real-time interaction, adaptive inference, and dynamic learning mechanisms.

## Architecture

<div align="center">
  <img src="https://raw.githubusercontent.com/nbursa/smart-chatbot-app/main/docs/images/full_architecture.png" width="500"/>
  <br/>
  <sub><i>Figure: Full architecture of Smart Chatbot App (frontend → backend → NLP)</i></sub>
</div>

## Features

- Frontend: Vue 3 SPA for interactive chat experience
- Middleware: Node.js API layer with MongoDB for message persistence
- Backend: Flask REST API with PyTorch models for NLP predictions
- Real-time adaptive learning using BERT and GPT-2
- Contextual memory with conversation history from MongoDB
- Custom meta-learning layer ("intuition") for self-adjustment

## Tech Stack

- **Frontend**: Vue 3, Axios, Tailwind CSS
- **Middleware Server**: Node.js (Express), TypeScript, ts-node, MongoDB (via Mongoose)
- **ML Backend**: Flask, PyTorch, Transformers (HuggingFace), scikit-learn
- **Model**: BERT with custom Intuition Layer + GPT-2 for generation
- **Others**: Python, JavaScript, HTML5, CSS3

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nbursa/smart-chatbot-app.git
```

### 2. Start the ML Backend (Flask + PyTorch)

```bash
cd api
pip install -r requirements.txt
python app.py
```

### 3. Start the Middleware Server (Node.js + Express)

```bash
cd server
npm install
npm run dev
```

### 4. Start the Frontend (Vue 3 + Vite)

```bash
cd client
npm install
npm run dev
```

## Adaptive AI Layer with DIAN (Dynamic Input Activation Network)

The `smart-chatbot-app` leverages a hybrid AI architecture that includes intent classification, generative modeling, and an experimental meta-learning component called **DIAN – Dynamic Input Activation Network**.

### What is DIAN?

DIAN is a novel architecture component designed by Nenad Bursać. It enables the model to adjust its internal layer behaviors during inference based on dynamic input signals. The mechanism simulates an _intuition layer_—a differentiable module that compares current activations to previously learned expectations and gradually updates weights and modulation coefficients.

See concept repo: [Dynamic Input Activation Network](https://github.com/nbursa/Dynamic-Input-Activation-Network)

### How it's implemented here

In `smart-chatbot-app`, DIAN is integrated via a custom PyTorch module inside the `BERTIntuitionModel`, which:

- Receives tokenized input from `BERT-base`
- Feeds the pooled output into `IntuitionNN` – a stacked feedforward network with optional gated skip connections and introspective adjustment layers
- During each forward pass, calculates a layer-wise difference and updates an `intuition_coefficients` tensor to reflect model confidence in its predictions
- Supports online updates via lightweight backpropagation and coefficient refinement

#### BERT + IntuitionNN Model Code

![BERTIntuitionModel Code](https://raw.githubusercontent.com/nbursa/smart-chatbot-app/main/docs/images/bert_intuition_model.png)

### Key Features

- **Online Adaptation**: The model updates itself after each prediction with minimal overhead
- **Introspection**: Intuition coefficients represent a soft measure of internal model alignment
- **Meta-awareness Simulation**: The system mimics a sense of “confidence” in its own inference process

#### compare_and_adjust Logic

![Compare and Adjust](https://raw.githubusercontent.com/nbursa/smart-chatbot-app/main/docs/images/compare_and_adjust.png)

### Purpose and Use Case

This chatbot system is not just a conversational agent—it's an **interactive AI research sandbox**. DIAN allows researchers and developers to:

- Experiment with real-time learning
- Explore adaptive layer tuning
- Analyze intuition-driven behavior across user inputs
- Visualize evolving model dynamics (via logs and plots)

> ❗ DIAN is still experimental. While promising, it is best suited for sandbox use, research, or internal tooling—not high-stakes production environments.

### DIAN Inference Flow

<div align="center">
  <img src="https://raw.githubusercontent.com/nbursa/smart-chatbot-app/main/docs/images/DIAN_inference_flow.png" width="500"/>
  <br/>
  <sub><i>Figure: DIAN inference process with intuition-driven adaptation.</i></sub>
</div>

## License

This project is open-source and available under the MIT License.

---

Built by Nenad Bursać.
