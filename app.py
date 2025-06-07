import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def extract_pdf_text(pdf_files):
    """Extract text from multiple PDFs with page numbers"""
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page_num, page in enumerate(pdf_reader.pages):
            if page_text := page.extract_text():
                text += f"--- Page {page_num+1} ---\n{page_text}\n\n"
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("pdf_faiss_index")
    return vector_store

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
    st.caption("Upload PDFs, ask questions, get instant answers")
    
    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.qa_chain = None
        st.session_state.quota_warning = False
        st.session_state.model_name = "gemini-1.5-flash"
    
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
                
                st.write("✂️ Splitting text into chunks...")
                text_chunks = chunk_text(raw_text)
                
                st.write("🧠 Creating knowledge base...")
                create_vector_store(text_chunks)
                
                st.session_state.processed = True
                st.session_state.quota_warning = False
                st.success("✅ PDFs processed successfully! You can now ask questions.")
                
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
    if st.session_state.processed:
        if not st.session_state.qa_chain:
            try:
                st.session_state.qa_chain = setup_qa_chain(st.session_state.model_name)
            except Exception as e:
                st.error(f"⚠️ Error initializing AI system: {str(e)}")
                st.error("Please ensure you're using the latest libraries: pip install -U langchain-google-genai")
                st.stop()
        
        user_query = st.chat_input("Ask about your PDFs...")
        if user_query:
            st.chat_message("user").write(user_query)
            
            with st.spinner("🔍 Searching documents..."):
                try:
                    start_time = time.time()
                    response = st.session_state.qa_chain({"query": user_query})
                    processing_time = time.time() - start_time
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(response["result"])
                        st.caption(f"⏱️ Response time: {processing_time:.1f} seconds | Model: {st.session_state.model_name}")
                        
                        # Show source pages
                        if response["source_documents"]:
                            with st.expander("🔍 Source Information"):
                                for i, doc in enumerate(response["source_documents"]):
                                    page_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    st.caption(f"**Source {i+1}**")
                                    st.info(page_content)
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        st.error("⚠️ API Quota Exceeded - You've hit the free tier limits")
                        st.error("Please try again later or upgrade your Google Cloud account")
                        st.session_state.quota_warning = True
                    else:
                        st.error(f"❌ Error processing your question: {str(e)}")
    
    # Quota warning display
    if st.session_state.get("quota_warning", False):
        st.warning("📢 You've exceeded your free tier quota. Here are your options:")
        st.markdown("""
        1. **⏳ Wait 1 minute** - Free tier resets every minute
        2. **💳 Upgrade your account** - [Google AI Studio Pricing](https://ai.google.dev/pricing)
        3. **🔑 Use a different API key** - If you have multiple projects
        4. **📉 Reduce usage** - Ask fewer questions or switch to Flash model
        """)
    
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
    #...