# 🤝 Contributing to Smart Chatbot App

Thank you for your interest in contributing to **Smart Chatbot App**! This project is an experimental AI system combining intent classification, generative dialogue, and adaptive neural components. All contributions—whether code, documentation, bug reports, or research ideas—are welcome.

---

## 🛠 How to Contribute

### 1. Fork the Repository

Click the "Fork" button at the top-right of this page to create your own copy of the repo.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/smart-chatbot-app.git
cd smart-chatbot-app
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

Update or add functionality in one of the layers:

- `client/` for frontend (Vue)
- `server/` for middleware (Node/Express)
- `api/` for ML backend (Flask + PyTorch)

### 5. Commit Changes

```bash
git commit -m "feat: description of your change"
```

### 6. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then go to GitHub and open a pull request against the `main` branch.

---

## 📦 Project Structure

```
smart-chatbot-app/
├── client/         # Vue 3 frontend
├── server/         # Node.js middleware + MongoDB
├── model/          # Flask + PyTorch ML backend
    └── data/       # Logs and training artifacts
├── docs/           # Architecture and AI design documents
```

---

## 🧪 Testing and Linting

Please test your changes and follow clean code practices. Linting rules are coming soon.

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Built with ❤️ by Nenad Bursać.
