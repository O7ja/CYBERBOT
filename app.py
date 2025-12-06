import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import glob

# Page configuration
st.set_page_config(
    page_title="CYBERBOT",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI Driven RAG CYBERBOT")
st.write("Ask questions about your PDF documents!")

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar configuration
with st.sidebar:
    st.header(" Configuration")
    
    pdf_folder = "C:/Users/DELL/Desktop/New folder (2)/souces"
    vector_store_path = "faiss_vector_store"
    
    st.subheader("LLM Settings")
    model_name = st.selectbox(
        "Select LLM Model",
        ["llama3.2:3b", "llama2-uncensored"],
        help="Choose the language model for responses"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more deterministic"
    )
    
    top_k = st.slider(
        "Number of Context Documents (k)",
        min_value=1,
        max_value=10,
        value=3,
        help="How many PDF chunks to retrieve for context"
    )
    
    st.subheader("System Status")
    
    # Check if PDFs exist
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    st.metric("PDFs Found", len(pdf_files))
    
    # Check if vector store exists
    if os.path.exists(vector_store_path):
        st.success("✓ Vector Store Ready")
    else:
        st.warning(" Vector Store Not Found")
    
    # Initialize RAG system
    if st.button("🔄 Initialize RAG System", use_container_width=True):
        with st.spinner("Loading components..."):
            try:
                # Load embeddings
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                st.write("✓ Embeddings loaded")
                
                # Load vector store
                if os.path.exists(vector_store_path):
                    st.session_state.vector_store = FAISS.load_local(
                        vector_store_path,
                        st.session_state.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    st.write("✓ Vector store loaded")
                else:
                    st.error("Vector store not found. Please create it first.")
                    st.stop()
                
                
                llm = OllamaLLM(
                    model=model_name,
                    base_url="http://localhost:11434",
                    temperature=temperature
                )
                
                
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": top_k}
                )
                
                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""Answer the question based on the context provided. Be specific and cite relevant information from the documents.

Context: {context}

Question: {question}

Answer:"""
                )
                
                def format_docs(docs):
                    return "\n\n---\n\n".join([doc.page_content for doc in docs])
                
                st.session_state.rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                )
                
                st.success(" RAG System Ready!")
                
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(" Chat with Your Documents")
    
    if st.session_state.rag_chain is None:
        st.info(" Click 'Initialize RAG System' in the sidebar to get started")
    else:
        # Chat input
        user_question = st.text_input(
            "Ask a question:",
            placeholder="What is cryptography?",
            key="user_input"
        )
        
        if user_question:
            with st.spinner("Retrieving documents and generating response..."):
                try:
                    response = st.session_state.rag_chain.invoke(user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response
                    })
                    
                    # Display response
                    st.write("---")
                    st.subheader("Answer:")
                    st.write(response)
                    st.write("---")
                    st.caption("*This answer is based on your PDF sources*")
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

with col2:
    st.subheader("Chat History")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"Q{i}: {chat['question'][:40]}..."):
                st.write(f"**Q:** {chat['question']}")
                st.write(f"**A:** {chat['answer']}")
    else:
        st.info("No chat history yet")
    
    if st.button(" Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.divider()
st.caption(" All processing is local - no data sent to external servers")
