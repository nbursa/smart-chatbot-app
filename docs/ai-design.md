# AI Design – Smart Chatbot App

This document explores the machine learning logic and adaptive mechanisms used in the Smart Chatbot App. The system blends classical intent classification with generative NLP and an experimental meta-learning layer (DIAN).

---

## Goals

- Combine BERT-based intent classification with GPT-2 response generation
- Integrate a dynamic adaptation mechanism to simulate meta-awareness
- Store and use conversational history from MongoDB for better context
- Enable real-time learning and introspection

---

## Architecture Overview

### Models

- **BERTIntuitionModel**:

  - Uses `bert-base-uncased` as encoder
  - Outputs passed to `IntuitionNN`, a multilayer feedforward network
  - Classification logits + intuition outputs are returned

- **GPT-2 (gpt2)**:
  - Used for free-form response generation based on prompt context
  - Prompt includes static instructions + recent conversation context

---

## Intuition Engine (DIAN)

DIAN (Dynamic Input Activation Network) is a proprietary research prototype designed by Nenad Bursać. It enables adaptive control over model internals by comparing learned expectations with real-time input signals.

### Core Principles

- Each layer learns its own expected activation ("intuition vector")
- Activations are compared in real-time with previous state
- Coefficients (confidence weights) are updated to reflect deviation
- Model evolves its behavior without full retraining

---

## How it works

1. Input is processed via BERT and pooled
2. `IntuitionNN` processes the pooled output through several layers
3. Intuition output is computed with `sigmoid(linear(x)) * coeffs`
4. A difference term is computed between activations and intuition
5. Online weight update is triggered
6. The model updates `intuition_coefficients` accordingly

---

## Logging & Visualization

- Logs every interaction to `data/conversation_logs/*.json`
- Each log contains:
  - user text
  - ai response
  - intent index
  - timestamp
- Visualization script (`plot_loss.py`) plots loss and intuition evolution

---

## Research Use

Ideal for:

- Studying intuition-influenced model updates
- Visualizing layer-wise behavioral change
- Exploring how GPT output varies with context history
- Benchmarking meta-adaptive NLP systems

---

## Limitations

- Not session-isolated (global context)
- Online updates may destabilize long-term performance
- DIAN is heuristic and still under research

---

## Future Work

- Add attention monitoring over conversation context
- Pre-train DIAN layer on synthetic intuition objectives
- Improve separation between base model and meta-layer

---

Built by Nenad Bursać • Concept Author of DIAN
