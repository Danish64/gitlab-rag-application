# GitLab RAG Application

A simple question-answering app that helps you find information from GitLab's People Group handbook.

## What is this?

This app lets you ask questions in plain English about GitLab's HR policies, onboarding, time off, learning programs, and more. Instead of digging through dozens of handbook pages, just ask your question and get an answer.

It uses RAG (Retrieval-Augmented Generation) which basically means:
1. Your question gets matched to relevant handbook sections
2. Those sections are sent to an AI model
3. The AI gives you an answer based on the actual handbook content

## How it works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  You ask a  │ --> │  Find best  │ --> │  Send to    │ --> │  Get your   │
│  question   │     │  matches    │     │  OpenAI     │     │  answer     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Project structure

```
├── app/              # Gradio web interface
├── api/              # FastAPI backend
├── src/              # Core logic (embeddings, retrieval, RAG)
├── documents/        # GitLab handbook markdown files
├── tests/            # Tests
└── eval_data/        # Evaluation data
```

## Setup

1. Clone the repo
   ```
   git clone https://github.com/Danish64/gitlab-rag-application.git
   cd gitlab-rag-application
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Add your OpenAI API key

   Create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Run the app

Start the Gradio interface:
```
python app/gradio_app.py
```

Or run the API server:
```
uvicorn api.main:app --reload
```

## Example questions

- "What is the paid time off policy?"
- "How do I request a relocation?"
- "What are the onboarding steps for new hires?"
- "How does the promotion process work?"

## Build Vector Database

The vector database (`chroma_db/`) is not included in git. You need to build it:

```bash
cd src && python embeddings.py
```

This will:
- Load 332 markdown documents from `documents/`
- Chunk them into ~6,500 pieces
- Embed using OpenAI `text-embedding-3-small`
- Store in ChromaDB at `chroma_db/`

### Rebuilding from scratch

```bash
# Option 1: Delete and rebuild
rm -rf chroma_db/
cd src && python embeddings.py

# Option 2: Force rebuild
cd src && python -c "from embeddings import create_vectorstore; create_vectorstore(force_rebuild=True)"
```

## Visualize Embeddings

```bash
cd src && python visualize.py
# Opens embedding_visualization.html in browser
```

## Tech stack

- **LangChain** - for building the RAG pipeline
- **ChromaDB** - vector database to store document embeddings
- **OpenAI** - for embeddings and chat completions
- **FastAPI** - API backend
- **Gradio** - simple web UI
