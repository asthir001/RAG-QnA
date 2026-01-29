# Conversational RAG with PDF Uploads and Chat History

This project is a sophisticated Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and engage in context-aware conversations. Built with **LangChain**, **Groq (Llama 3.3)**, and **HuggingFace**, the system intelligently reformulates questions based on chat history to provide accurate, document-specific answers.



## Project Structure

- `app.py`: The core Streamlit application containing the RAG pipeline, state management, and UI.
- `.env`: Local environment file for sensitive API keys.
- `.gitignore`: Configured to exclude IDE settings, virtual environments, and temporary files.

## Features

- **Contextual Awareness**: Utilizes a `history_aware_retriever` to understand follow-up questions (e.g., "What does it say about X?" followed by "Can you elaborate on that?").
- **Real-time PDF Processing**: Uploaded PDFs are split using `RecursiveCharacterTextSplitter` and indexed on-the-fly.
- **Vector Search**: Uses **ChromaDB** with **HuggingFace `all-MiniLM-L6-v2`** embeddings for high-accuracy document retrieval.
- **Llama 3.3 Integration**: Powered by the `llama-3.3-70b-versatile` model via Groq for fast, human-like responses.
- **Session-Based Memory**: Supports custom `Session ID` inputs to maintain different conversation threads.

## Getting Started

### Prerequisites

- Python 3.9 or higher.
- A **Groq API Key** and a **HuggingFace Token**.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ashtir001/RAG-QnA.git
   cd Conversational_RAG_PDF
   ```


2. **Install all required libraries**:
```bash
   pip install -r requirements
```

3. **Configure Environment Variables**:
   Create a .env file in your root folder and paste your keys: GROQ_API_KEY=your_groq_key_here HUGGING_FACE=your_huggingface_token_here

4. **Run the App**:
   ```bash
   streamlit run app.py
    ```
   
## Usage
**Set Session ID**: Provide a unique ID for your chat session.

**Upload PDF**: Choose a PDF file from your computer. The app will index it.

**Ask Questions**: Type your question in the input field.

**Contextual History**: Ask follow-up questions naturally; the assistant remembers the context.

## Technologies Used
**LLM**: ChatGroq (Llama-3.3-70b-versatile)

**Embeddings**: HuggingFace (all-MiniLM-L6-v2)

**Vector Database**: Chroma

**Orchestration**: LangChain

**Frontend**: Streamlit

## License
MIT
