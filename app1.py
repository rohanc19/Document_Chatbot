import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Set up API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit app configuration
st.set_page_config(page_title="Gemma Model Document Q&A", layout="wide")
st.title("Gemma Model Document Q&A")

# Initialize session state
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

# LLM and prompt setup
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

llm = get_llm()

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the questions.

<context>
{context}
</context>

Question: {input}
""")

# Vector embedding function
def create_vector_store():
    with st.spinner("Creating Vector Store..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(final_documents, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.vector_store_ready = True
    st.success("Vector Store is ready!")

# Sidebar for vector store creation
with st.sidebar:
    st.header("Vector Store Setup")
    if st.button("Create Vector Store"):
        create_vector_store()

# Main app layout
col1, col2 = st.columns([2, 1])

with col1:
    user_question = st.text_input("What do you want to ask about the documents?")
    if st.button("Ask") and user_question:
        if not st.session_state.vector_store_ready:
            st.error("Please create the Vector Store first!")
        else:
            with st.spinner("Searching for an answer..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start_time = time.time()
                response = retrieval_chain.invoke({"input": user_question})
                end_time = time.time()
                
                st.markdown("### Answer:")
                st.write(response["answer"])
                st.info(f"Response time: {end_time - start_time:.2f} seconds")

with col2:
    if "vector_store_ready" in st.session_state and st.session_state.vector_store_ready:
        st.success("Vector Store is ready for queries!")
    else:
        st.warning("Vector Store not created yet. Use the sidebar to create it.")

    if "answer" in locals():
        with st.expander("Document Similarity Search Results", expanded=False):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Groq")