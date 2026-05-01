# 📘 Enterprise-Grade RAG Chatbot  
**Developed by:** K. Kanishthika  

A Retrieval-Augmented Generation (RAG) chatbot built using **LangChain, ChromaDB, and Groq’s Llama 3**.  
It allows users to query a custom dataset and receive **accurate, context-aware responses without hallucinations**.

---

## 📸 Demo / Preview  
<!-- Add your screenshot below -->
<img width="1893" height="966" alt="image" src="https://github.com/user-attachments/assets/dcfc7dc7-9b65-468a-a8c3-056aaef99904" />


---

## 🏗️ Architecture Overview  

This chatbot follows a **two-stage pipeline**:

### 1️⃣ The Librarian (Retrieval)
- Converts user query into vectors using:
  - `sentence-transformers/all-MiniLM-L6-v2`
- Searches **ChromaDB vector database**
- Retrieves **top 3 relevant chunks**

### 2️⃣ The Writer (Generation)
- Combines:
  - User query + retrieved context
- Sends to **Groq API (Llama 3)** using LangChain (LCEL)

### 3️⃣ Final Output
- Generates a **context-aware answer**
- Displays via **Gradio UI**

> 🔒 The model only answers from retrieved context → **No hallucination**

---

## 📚 Dataset  

- Source: **2022 State of the Union Address**
- File: `data.txt`

### Processing:
- Split using `RecursiveCharacterTextSplitter`
- Chunk size: **500**
- Overlap: **50**

✔ Preserves sentence continuity  
✔ Improves retrieval accuracy  

---

## 🚧 Learning Journey & Challenges  

### 🔹 Dependency Conflicts
- Faced version issues with:
  - `gradio`
  - `langchain-huggingface`
  - `huggingface-hub`
- Solved using updated **LangChain LCEL architecture**

---

### 🔹 Hugging Face API Issues
- Tried models like:
  - Mistral
  - Zephyr  
- Encountered:
  - `400 Bad Request`
  - Provider routing issues (`featherless-ai`)

---

### 🔹 Security Mistake
- Initially hardcoded API keys ❌  
- Fixed using:
  - `.env` file
  - `python-dotenv` ✅  

---

## ⚡ Why Groq?  

Switched to Groq due to instability in Hugging Face inference APIs.

### Benefits:
- Ultra-fast inference (LPUs)
- Better instruction following
- Reliable API

### Model Used:
- `llama-3.3-70b-versatile`

---

## 🛠️ Tech Stack  

| Component | Tool |
|----------|-----|
| Framework | LangChain (LCEL) |
| LLM | Llama 3 (Groq API) |
| Embeddings | Hugging Face |
| Vector DB | ChromaDB |
| UI | Gradio |
| Env Mgmt | python-dotenv |

---

## How to Run Locally  

### 1️⃣ Clone Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash 
pip install langchain langchain-community langchain-huggingface langchain-groq gradio chromadb sentence-transformers python-dotenv
```

### 4️⃣ Setup Environment Variables
Create a .env file in the root directory:
```bash
GROQ_API_KEY=your_api_key_here
```
### 5️⃣ Run Application
```bash
python app.py
```
### 6️⃣ Access UI
http://127.0.0.1:7860

### 🎯 Key Features

- ✔ Context-aware responses
- ✔ Zero hallucination design
- ✔ Fast inference via Groq
- ✔ Modular RAG pipeline
- ✔ Clean Gradio UI

### 📌 Future Improvements
- Add multi-document support
- Deploy on cloud (Render / Hugging Face Spaces)
- Add chat history memory
- Improve UI/UX
### ⭐ Acknowledgment

-Built as part of hands-on learning in:

  - Retrieval-Augmented Generation (RAG)
  - LLM system design
  - Production-ready AI pipelines

### Learning Phase!!!

