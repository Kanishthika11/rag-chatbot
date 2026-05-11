import importlib
import os
import threading
from pathlib import Path

import gradio as gr
import pyttsx3
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

# --- 1. VOICE ENGINE ---
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# --- 2. RAG PIPELINE ---
LOCAL_PDFS = sorted(
    [
        p
        for p in BASE_DIR.iterdir()
        if p.suffix.lower() == ".pdf" and p.name.lower().startswith("gst_")
    ],
    key=lambda p: p.name.lower(),
)


def get_pdf_loader():
    candidates = [
        "langchain_community.document_loaders",
        "langchain_community.document_loaders.pdf",
        "langchain.document_loaders",
        "langchain.document_loaders.pdf",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        loader = getattr(module, "PyPDFLoader", None)
        if loader is not None:
            return loader

        loader = getattr(module, "UnstructuredPDFLoader", None)
        if loader is not None:
            return loader

    raise ImportError(
        "PDF loading requires additional packages or a supported LangChain loader path. "
        "Install 'pypdf' or 'unstructured' and then rerun the app. Example: pip install pypdf unstructured"
    )


def load_pdf_documents(pdf_path: Path):
    loader_cls = get_pdf_loader()
    print(f"Loading PDF: {pdf_path}")
    return loader_cls(str(pdf_path)).load()


def load_documents():
    documents = []
    data_txt = BASE_DIR / "data.txt"

    if data_txt.exists():
        print("Loading text data from data.txt...")
        docs = TextLoader(str(data_txt), encoding="utf-8").load()
        print(f"Loaded {len(docs)} document chunks from data.txt")
        documents.extend(docs)
    else:
        print("Warning: data.txt not found. Only PDF sources will be used.")

    if not LOCAL_PDFS:
        print("Warning: No GST PDF files found in the project folder.")

    for pdf_path in LOCAL_PDFS:
        if pdf_path.exists():
            docs = load_pdf_documents(pdf_path)
            print(f"Loaded {len(docs)} document chunks from {pdf_path.name}")
            documents.extend(docs)
        else:
            print(f"Warning: Local PDF not found: {pdf_path}")

    if not documents:
        raise ValueError(
            "No documents were loaded. Add data.txt or PDF files and restart the app."
        )

    return documents


def setup_rag_pipeline():
    print("1/4 Loading and splitting data...")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print("2/4 Building the local vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    print("3/4 Connecting to Groq AI Engine...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=300,
    )

    print("4/4 Assembling the RAG pipeline...")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use ONLY the context below to answer.\n"
         "If the answer is not in the context, say: "
         "'I cannot answer this based on the provided text.'\n\n"
         "Context:\n{context}"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

print("🔧 Initializing RAG pipeline...")
rag_chain = setup_rag_pipeline()

# --- 3. CHAT FUNCTION ---
# Stores the last answer so the speaker button can read it
last_answer = {"text": ""}

def chat_interface(message, history):
    try:
        answer = rag_chain.invoke(message)
        last_answer["text"] = answer
        return answer
    except Exception as e:
        error_msg = f"🚨 Error: {str(e)}"
        last_answer["text"] = error_msg
        return error_msg

def speak_last_answer():
    text = last_answer.get("text", "")
    if text:
        speak(text)

# --- 4. GRADIO UI ---
print("\n✅ All systems go! Launching UI...")

with gr.Blocks(title="🎙️ Voice RAG Chatbot") as demo:
    gr.Markdown("""
        <div style="text-align: center;">
            <h1>🎙️ Voice-Enabled RAG Chatbot</h1>
            <p>Ask questions about your document. Click 🔊 to hear the last answer read aloud.</p>
        </div>
        """)
    chatbot = gr.ChatInterface(
        fn=chat_interface,
        chatbot=gr.Chatbot(height=420),
        textbox=gr.Textbox(
            placeholder="Ask a question about the document...",
            container=False,
            scale=7,
            submit_btn="➤",
        ),
        title=None,         # Title already set above
        description=None,
    )

    with gr.Row():
        speak_btn = gr.Button("🔊 Read Last Answer Aloud", variant="secondary", scale=1)

    speak_btn.click(fn=speak_last_answer, inputs=[], outputs=[])

if __name__ == "__main__":
    demo.launch()