import os
import gradio as gr
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

def setup_rag_pipeline():
    print("1/4 Loading and splitting data...")
    loader = TextLoader("daata.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("2/4 Building the local vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("3/4 Connecting to Groq AI Engine...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Free, fast, reliable
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

def chat_interface(message, history):
    try:
        answer = rag_chain.invoke(message)
        return answer
    except Exception as e:
        return f"🚨 Error: {str(e)}"

print("\n✅ All systems go! Launching UI...")
demo = gr.ChatInterface(
    fn=chat_interface,
    title="RAG Chatbot",
    description="Ask questions about the loaded document."
)

if __name__ == "__main__":
    demo.launch()