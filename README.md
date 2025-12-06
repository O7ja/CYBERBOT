# RAG Chatbot - Streamlit Frontend

A beautiful, interactive Streamlit frontend for your RAG (Retrieval-Augmented Generation) chatbot that answers questions based on your cryptography PDF documents.

## Features

✨ **Interactive Chat Interface**
- Ask questions about your PDFs
- Get answers grounded in your document sources
- View chat history

⚙️ **Configurable Settings**
- Choose between LLM models (llama3.2:3b, llama2-uncensored)
- Adjust temperature for response creativity
- Control number of context documents retrieved

📊 **System Monitoring**
- Real-time status of PDFs and vector store
- One-click RAG system initialization
- Chat history tracking

🔐 **Local & Private**
- All processing happens locally
- No data sent to external servers
- Uses local Ollama LLM and embeddings

## Installation

### Prerequisites
- Python 3.10+
- Ollama installed and running locally
- Required LLM models pulled: `ollama pull llama3.2:3b`

### Setup Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running:**
   ```bash
   # In a separate terminal
   ollama serve
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   - Open your browser to `http://localhost:8501`
   - Click "Initialize RAG System" in the sidebar
   - Start asking questions!

## Project Structure

```
├── app.py                          # Streamlit frontend
├── projecy.ipynb                   # Development notebook (RAG setup)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── souces/                         # Your PDF documents
│   ├── Serious_Cryptography.pdf
│   ├── cryptography_book_1.pdf
│   └── cryptography_book_2.pdf
└── faiss_vector_store/             # FAISS vector store (auto-created)
    ├── index.faiss
    ├── index.pkl
    └── docstore.pkl
```

## How It Works

1. **PDF Loading**: Loads documents from `souces/` folder using PyMuPDF
2. **Text Chunking**: Splits documents into 2000-char chunks for semantic coherence
3. **Embeddings**: Creates 384-dimensional vectors using BGE embeddings
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: Retrieves top-k most relevant chunks for each query
6. **Generation**: Passes retrieved context to local Ollama LLM for answer generation

## Configuration

### LLM Models
- **llama3.2:3b** (Recommended): Fast, lightweight, fits 4.5GB RAM
- **llama2-uncensored**: Alternative option

To add more models, pull them with Ollama:
```bash
ollama pull model_name
```

### Embedding Model
Uses BAAI/bge-small-en-v1.5 (384-dimensional) for semantic search

### Vector Store
Stored at `faiss_vector_store/` with automatic disk caching for instant loads

## Troubleshooting

**Issue: "Vector Store Not Found"**
- Run the notebook cells first to generate the vector store
- Check `faiss_vector_store/` directory exists

**Issue: "Connection refused to Ollama"**
- Make sure Ollama is running: `ollama serve` in a separate terminal
- Check localhost:11434 is accessible

**Issue: Slow response times**
- First query is slower (loads embedding model)
- Subsequent queries are faster
- Reduce `top_k` for fewer documents to process

**Issue: Out of Memory**
- llama3.2:3b requires ~3GB RAM
- Close other applications
- Reduce vector store size by re-chunking documents

## Example Queries

- "What is cryptography?"
- "Explain encryption methods"
- "What are the differences between symmetric and asymmetric encryption?"
- "How does AES work?"
- "What is public-key cryptography?"

## Performance Metrics

- **PDF Loading**: ~1-2 seconds for 3 documents
- **Vector Store Creation**: ~40 minutes (first time only, then instant)
- **Query Response**: 30-60 seconds (including embedding + retrieval + LLM generation)
- **Memory Usage**: ~4GB total (embeddings + LLM + vector store)

## Next Steps

- Add document source citations in responses
- Implement multi-document comparison
- Add export functionality for responses
- Support for custom PDFs via UI upload
- Fine-tuning for cryptography domain

## License

MIT License - Feel free to use and modify!
