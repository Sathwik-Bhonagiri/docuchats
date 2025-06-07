import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import time
import json
from datetime import datetime

# Initialize environment
def setup_environment():
    if not os.getenv("GOOGLE_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "Google API Key not found in environment variables!"
    os.environ["GOOGLE_API_KEY"] = api_key
    return api_key

# PDF Processing Functions
import fitz  # PyMuPDF

def extract_pdf_text(pdf_files):
    """Extract text from multiple PDFs using PyMuPDF"""
    text = ""
    for pdf_file in pdf_files:
        # Create a BytesIO object from the uploaded file
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"--- Page {page_num+1} ---\n{page.get_text()}\n\n"
    return text

def chunk_text(text, chunk_size=800, chunk_overlap=150):
    """Split text into manageable chunks with optimized size"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    """Create and save FAISS vector store safely"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("pdf_faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        st.error("This is usually caused by:")
        st.error("- Invalid Google API key")
        st.error("- PDFs with scanned images (non-text content)")
        st.error("- Large PDF files that exceed memory limits")
        return None

def load_vector_store(embeddings):
    """Load vector store with error handling and fallback"""
    try:
        # First try to load existing vector store
        return FAISS.load_local(
            "pdf_faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.warning(f"⚠️ Vector store loading failed: {str(e)}")
        
        # If we have text chunks in session state, try to recreate
        if "text_chunks" in st.session_state and st.session_state.text_chunks:
            st.info("Attempting to recreate vector store from text chunks...")
            return create_vector_store(st.session_state.text_chunks)
        
        st.error("Could not recover vector store. Please reprocess your PDFs.")
        return None

# QA System Setup
def setup_qa_chain(model_name="gemini-1.5-flash"):
    """Create retrieval-based QA system with safe loading"""
    prompt_template = """
    Answer the question concisely based only on the following context. 
    If the answer isn't in the context, say "I couldn't find that in the document."
    Keep answers brief - maximum 2-3 sentences.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    # Use the selected model
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        max_output_tokens=300
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load vector store with safe deserialization
    vector_store = FAISS.load_local(
        "pdf_faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Streamlit UI
def main():
    api_key = setup_environment()
    
    st.set_page_config(
        page_title="PDF Chat Assistant", 
        layout="centered",
        page_icon="📄"
    )
    st.title("📄 Chat with PDFs using Gemini")
    
    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.qa_chain = None
        st.session_state.quota_warning = False
        st.session_state.model_name = "gemini-1.5-flash"
        st.session_state.chat_history = []
        st.session_state.processing_metrics = {}
    
    # Sidebar for configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        # Model selection
        model_option = st.selectbox(
            "Select Gemini Model",
            options=[
                ("🚀 Flash (Fast & Efficient)", "gemini-1.5-flash"),
                ("⚖️ Pro (Balanced)", "gemini-1.0-pro"),
                ("🧠 Advanced (Long Context)", "gemini-1.5-pro-latest")
            ],
            format_func=lambda x: x[0],
            index=0
        )
        st.session_state.model_name = model_option[1]
        
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader(
            "Select PDF documents", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        process_button = st.button("Process PDFs", disabled=not pdf_files)
        if process_button and pdf_files:
            with st.status("Processing documents..."):
                # Extract and process text
                st.write("📖 Reading PDF content...")
                raw_text = extract_pdf_text(pdf_files)
                
                # Calculate processing metrics
                char_count = len(raw_text)
                page_count = sum(1 for pdf in pdf_files for _ in PdfReader(pdf).pages)
                file_count = len(pdf_files)
                
                st.write("✂️ Splitting text into chunks...")
                text_chunks = chunk_text(raw_text)
                chunk_count = len(text_chunks)
                
                st.write("🧠 Creating knowledge base...")
                create_vector_store(text_chunks)
                
                # Store processing metrics
                st.session_state.processing_metrics = {
                    "characters": char_count,
                    "chunks": chunk_count,
                    "pages": page_count,
                    "files": file_count,
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.processed = True
                st.session_state.quota_warning = False
                st.session_state.chat_history = []  # Clear chat history on new processing
                st.success("✅ PDFs processed successfully! You can now ask questions.")
        
        # Display processing metrics
        if st.session_state.processed and st.session_state.processing_metrics:
            st.subheader("Processing Metrics")
            metrics = st.session_state.processing_metrics
            st.metric("Total Files", metrics["files"])
            st.metric("Total Pages", metrics["pages"])
            st.metric("Characters Processed", f"{metrics['characters']:,}")
            st.metric("Text Chunks Created", metrics["chunks"])
            st.caption(f"Processed at: {metrics['processed_at']}")
                
        # Reset button
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()
                
        # Quota information
        st.markdown("---")
        st.info("**Free Tier Limitations:**\n"
                "- 60 requests/minute\n"
                "- 1,500 requests/day\n\n"
                "Upgrade at [Google AI Studio](https://aistudio.google.com/)")

    # Main chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        st.subheader("Chat History")
        
        if not st.session_state.chat_history:
            st.info("💬 Your chat history will appear here. Start by asking a question!")
        
        # Display chat history in reverse order (newest at bottom)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                # User question
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(chat["question"])
                    st.caption(chat["timestamp"])
                
                # Assistant response
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(chat["answer"])
                    
                    # Source information
                    if chat.get("sources"):
                        st.markdown("**Sources used:**")
                        for source in chat["sources"]:
                            st.caption(f"📄 {source}")
                    
                    st.caption(f"⏱️ Response time: {chat['response_time']:.1f}s | Model: {st.session_state.model_name}")
    
    # Question input at bottom
    user_query = st.chat_input("Ask about your PDFs...", key="chat_input")
    
    if user_query and st.session_state.processed:
        # Add user question to history immediately
        new_chat = {
            "question": user_query,
            "answer": "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response_time": 0,
            "sources": []
        }
        st.session_state.chat_history.append(new_chat)
        
        # Process the question
        if not st.session_state.qa_chain:
            try:
                st.session_state.qa_chain = setup_qa_chain(st.session_state.model_name)
            except Exception as e:
                st.error(f"⚠️ Error initializing AI system: {str(e)}")
                st.error("Please ensure you're using the latest libraries: pip install -U langchain-google-genai")
                st.stop()
        
        with st.spinner("🔍 Searching documents..."):
            try:
                start_time = time.time()
                response = st.session_state.qa_chain({"query": user_query})
                processing_time = time.time() - start_time
                
                # Update the latest chat entry
                st.session_state.chat_history[-1]["answer"] = response["result"]
                st.session_state.chat_history[-1]["response_time"] = processing_time
                
                # Extract sources
                sources = set()
                for doc in response["source_documents"]:
                    source = f"Page {doc.metadata.get('page', 'N/A')} - {doc.page_content[:100]}..."
                    sources.add(source)
                st.session_state.chat_history[-1]["sources"] = list(sources)
                
                # Rerun to update chat display
                st.rerun()
                
            except Exception as e:
                st.session_state.chat_history[-1]["answer"] = f"Error: {str(e)}"
                if "quota" in str(e).lower() or "429" in str(e):
                    st.session_state.chat_history[-1]["answer"] = "⚠️ API Quota Exceeded - Please try again later"
                    st.session_state.quota_warning = True
                st.rerun()
    
    # Initial instructions
    if not st.session_state.processed:
        st.info("👋 Welcome! To get started:")
        st.markdown("""
        1. 👉 **Upload PDF files** in the sidebar
        2. 🚀 Click **Process PDFs** to build the knowledge base
        3. 💬 Start asking questions in the chat
        
        For best results:
        - Use the Flash model for quick responses
        - Keep questions focused on document content
        - Process only necessary documents
        """)
        st.image("https://images.unsplash.com/photo-1497636577773-f1231844b336?auto=format&fit=crop&w=600", 
                 caption="AI Document Analysis")

if __name__ == "__main__":
    main()