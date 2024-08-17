# Gemma Model Document Q&A App

## Overview

This Streamlit application leverages the power of the Gemma language model to provide a question-answering system based on PDF documents. The app uses LangChain for document processing and retrieval, Groq for language model inference, and Google's Generative AI for embeddings.

## Features

- PDF document loading and processing
- Vector store creation for efficient document retrieval
- Natural language querying of document content
- Interactive user interface with Streamlit

## Prerequisites

- Python 3.7+
- Groq API key
- Google API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gemma-document-qa.git
   cd gemma-document-qa
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Place your PDF documents in a folder named `us_census` in the project root directory.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

4. In the sidebar, click "Create Vector Store" to process the documents and create the vector store.

5. Once the vector store is ready, enter your question in the text input field and click "Ask" to get an answer based on the document content.

## How it Works

1. The app loads PDF documents from the specified directory.
2. It creates a vector store using Google's Generative AI embeddings for efficient document retrieval.
3. User queries are processed using the Gemma language model via Groq.
4. Relevant document sections are retrieved and combined to generate an answer.

## File Structure

```
gemma-document-qa/
│
├── app.py              # Main Streamlit application
├── .env                # Environment variables (API keys)
├── requirements.txt    # Python dependencies
├── us_census/          # Directory for PDF documents
└── README.md           # This file
```

## Dependencies

- streamlit
- python-dotenv
- langchain
- langchain-groq
- langchain-google-genai
- faiss-cpu
- pypdf

## Troubleshooting

- If you encounter any issues with API keys, ensure they are correctly set in your `.env` file.
- Make sure your PDF documents are placed in the `us_census` folder before creating the vector store.
- If you face performance issues, consider using a smaller subset of documents or adjusting the chunk size in the text splitter.

## Contributing

Contributions to improve the app are welcome. Please feel free to submit a Pull Request.

## Acknowledgments

- Streamlit for the amazing web app framework
- LangChain for document processing and chain operations
- Groq for providing access to the Gemma language model
- Google for the Generative AI embeddings
