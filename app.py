import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document, HumanMessage
import os
import time
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout decorator for API calls
def timeout_handler(seconds=15):
    """Timeout decorator to prevent hanging API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"API call timed out after {seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator

# Initialize environment
def setup_environment():
    # Check Streamlit secrets first
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        return st.secrets["GOOGLE_API_KEY"]
    
    # Then check environment variables
    if api_key := os.getenv("GOOGLE_API_KEY"):
        return api_key
    
    # Finally try .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if api_key := os.getenv("GOOGLE_API_KEY"):
            return api_key
    except:
        pass
    
    st.error("Google API Key not found! Add it to Streamlit secrets.")
    st.stop()

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

def create_vector_store(text_chunks, preferred_model=None):
    """Create FAISS vector store in memory"""
    def _get_embeddings_instance(preferred_model=None):
        # Allow explicit override via env var
        env_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
        candidates = []
        # preferred_model (from UI) should be tried first
        if preferred_model:
            candidates.append(preferred_model)
        if env_model:
            candidates.append(env_model)

        # Google Generative AI embedding models (not OpenAI models).
        # Try commonly available Google embedding model names first.
        candidates.extend([
            "models/text-embedding-004",  # Default Google embedding model
            "models/embedding-gecko-001",  # Gecko family (older clients)
            "textembedding-gecko-001",  # Alternate community name
            "embed-text-001"  # fallback alias
        ])

        last_err = None
        for model_name in candidates:
            # Skip obviously invalid placeholders
            if not isinstance(model_name, str) or "{" in model_name or "}" in model_name:
                continue

            # Try model as given, and if it doesn't start with required prefix, try prefixed variant too
            trials = [model_name]
            if not (isinstance(model_name, str) and (model_name.startswith("models/") or model_name.startswith("tunedModels/"))):
                trials.append(f"models/{model_name}")

            for trial in trials:
                try:
                    logger.info(f"Trying embedding model: {trial}")
                    if trial is None:
                        embeddings = GoogleGenerativeAIEmbeddings()
                    else:
                        embeddings = GoogleGenerativeAIEmbeddings(model=trial)
                except Exception as e:
                    logger.info(f"Embedding init failed for {trial}: {e}")
                    last_err = e
                    continue

                # Validate the model by calling a tiny embedding request; skip model if it fails
                try:
                    embeddings.embed_documents(["test"])
                    return embeddings
                except Exception as e:
                    logger.info(f"Embedding validation failed for {trial}: {e}")
                    last_err = e
                    continue

        # If we get here, no candidate worked
        logger.error("No valid embedding model found via Google Generative API.")
        # Inform the user but continue with a lightweight local fallback using TF-IDF
        st.warning("No valid Google embedding model found. Falling back to a local TF-IDF embedding implementation.")
        if last_err:
            st.info(f"Last error: {str(last_err)}")

        # Local TF-IDF fallback embeddings (lightweight, no GPU required)
        try:
            class TfidfEmbeddings:
                def __init__(self, max_features=1024):
                    self.vectorizer = TfidfVectorizer(max_features=max_features)
                    self._fitted = False

                def embed_documents(self, texts):
                    if not texts:
                        return []
                    if not self._fitted:
                        # Fit on the corpus
                        self.vectorizer.fit(texts)
                        self._fitted = True
                    arr = self.vectorizer.transform(texts).toarray()
                    return arr.tolist()

                def embed_query(self, text):
                    if not self._fitted:
                        # Fit on the single query to produce a vector (best-effort)
                        self.vectorizer.fit([text])
                        self._fitted = True
                    return self.vectorizer.transform([text]).toarray()[0].tolist()

                def __call__(self, texts):
                    return self.embed_documents(texts)

            logger.info("Using local TF-IDF embeddings as fallback.")
            return TfidfEmbeddings()
        except Exception as e:
            logger.error(f"TF-IDF fallback failed: {e}")
            st.error(f"All embedding methods failed. Last error: {e}")
            st.stop()

    embeddings = _get_embeddings_instance(preferred_model=preferred_model)

    # Coerce text_chunks into a list of strings (handle accidental single-string inputs)
    if isinstance(text_chunks, str):
        text_chunks = chunk_text(text_chunks)
    if not isinstance(text_chunks, (list, tuple)):
        try:
            text_chunks = list(text_chunks)
        except Exception:
            text_chunks = [str(text_chunks)]

    text_chunks = [str(t) for t in text_chunks]

    # Build Document objects so downstream chains always receive Documents
    docs = [Document(page_content=t) for t in text_chunks]
    
    # Create simple in-memory vector store
    class SimpleVectorStore:
        """Lightweight in-memory vector store for any embedding"""
        def __init__(self, documents, embeddings):
            self.documents = documents
            self.embeddings = embeddings
            # Pre-compute embeddings for all documents
            texts = [d.page_content for d in documents]
            self.doc_vectors = embeddings.embed_documents(texts)
        
        def as_retriever(self, search_kwargs=None):
            k = search_kwargs.get("k", 3) if search_kwargs else 3
            return SimpleRetriever(self.documents, self.doc_vectors, self.embeddings, k)
    
    class SimpleRetriever:
        """Simple retriever using semantic similarity"""
        def __init__(self, documents, doc_vectors, embeddings, k):
            self.documents = documents
            self.doc_vectors = doc_vectors
            self.embeddings = embeddings
            self.k = k
        
        def get_relevant_documents(self, query):
            """Retrieve k most similar documents"""
            # Embed the query
            query_vector = self.embeddings.embed_query(query)
            query_vector = np.array(query_vector)
            doc_vectors = np.array(self.doc_vectors)
            
            # Compute cosine similarity
            similarities = np.dot(doc_vectors, query_vector) / (
                np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-8
            )
            
            # Get top k documents
            top_indices = np.argsort(similarities)[-self.k:][::-1]
            return [self.documents[i] for i in top_indices]
    
    try:
        return SimpleVectorStore(docs, embeddings)
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        st.error(f"Failed to create vector store: {e}")
        st.stop()


# QA System Setup
def setup_qa_chain(vector_store, model_name="gemini-1.5-flash"):
    """Create retrieval-based QA system using a custom wrapper"""
    
    class SimpleQAChain:
        """Custom QA chain that avoids RetrievalQA chain issues"""
        def __init__(self, vector_store, model_name, prompt_template):
            self.vector_store = vector_store
            # Try to initialize the remote Chat model; fall back to a lightweight
            # local model when the Google API key is not available or init fails.
            try:
                self.model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.2,
                    max_output_tokens=300
                )
            except Exception as e:
                logger.warning(f"ChatGoogleGenerativeAI init failed, using local fallback: {e}")

                class LocalFallbackModel:
                    """Very small local 'LLM' that extracts the most relevant
                    sentences from the provided context as a best-effort answer.
                    This allows the app to function without external API keys.
                    """
                    def predict(self, prompt_text):
                        try:
                            import re
                            # Try to extract the context and question from the
                            # prompt template used elsewhere in this file.
                            m = re.search(r"Context:\s*(.*?)\n\nQuestion:\s*(.*?)\n\nAnswer:", prompt_text, re.S)
                            if m:
                                context = m.group(1)
                                question = m.group(2)
                            else:
                                # Fallback: treat whole prompt as context
                                context = prompt_text
                                question = ""

                            # Split context into candidate sentences/chunks
                            candidates = [s.strip() for s in re.split(r"[\.\n]+", context) if s.strip()]
                            if not candidates:
                                return "I couldn't find that in the document."

                            # Score candidates by simple keyword overlap with the question
                            q_words = set(w.lower() for w in re.findall(r"\w+", question))
                            scored = []
                            for c in candidates:
                                c_words = set(w.lower() for w in re.findall(r"\w+", c))
                                score = len(q_words & c_words)
                                scored.append((score, c))
                            scored.sort(reverse=True)

                            # If no overlap, return a brief instruction message
                            if scored[0][0] == 0:
                                # Return the first couple of sentences as context summary
                                return (candidates[0] + (" " + candidates[1] if len(candidates) > 1 else ""))

                            # Return the top 1-2 scoring sentences
                            top = [s for sc, s in scored[:2]]
                            return " ".join(top)
                        except Exception:
                            return "I couldn't find that in the document."

                self.model = LocalFallbackModel()
            self.prompt_template = prompt_template
            self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        def __call__(self, input_dict):
            """Handle dict input like RetrievalQA"""
            query = input_dict.get("query") if isinstance(input_dict, dict) else input_dict
            return self.answer(query)
        
        def run(self, query):
            """Handle .run() calls"""
            result = self.answer(query)
            return result.get("result", str(result))
        
        def answer(self, query):
            """Perform retrieval and generate answer"""
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Format context from documents
            context = "\n\n".join([d.page_content for d in docs])
            
            # Build full prompt
            full_prompt = self.prompt_template.format(context=context, question=query)
            
            # Get LLM response with fallback for quota errors and timeout
            response_text = None
            
            # Try the model with a timeout
            try:
                # Use timeout for API calls (15 seconds max)
                @timeout_handler(seconds=10)
                def invoke_model():
                    if hasattr(self.model, 'invoke'):
                        # For ChatGoogleGenerativeAI and other LangChain models
                        response = self.model.invoke([HumanMessage(content=full_prompt)])
                        return response.content
                    elif hasattr(self.model, 'predict'):
                        return self.model.predict(full_prompt)
                    else:
                        return str(self.model(full_prompt))
                
                response_text = invoke_model()
            except Exception as e:
                # If API fails (quota, network, timeout, etc.), use local fallback
                error_str = str(e).lower()
                logger.warning(f"Model invocation failed: {e}")
                
                # Use local fallback extraction
                try:
                    import re
                    m = re.search(r"Context:\s*(.*?)\n\nQuestion:\s*(.*?)\n\nAnswer:", full_prompt, re.S)
                    if m:
                        context_text = m.group(1)
                        question_text = m.group(2)
                    else:
                        context_text = full_prompt
                        question_text = ""
                    
                    candidates = [s.strip() for s in re.split(r"[\.\n]+", context_text) if s.strip()]
                    if not candidates:
                        response_text = "I couldn't find that in the document."
                    else:
                        q_words = set(w.lower() for w in re.findall(r"\w+", question_text))
                        scored = []
                        for c in candidates:
                            c_words = set(w.lower() for w in re.findall(r"\w+", c))
                            score = len(q_words & c_words)
                            scored.append((score, c))
                        scored.sort(reverse=True)
                        
                        if scored[0][0] == 0:
                            response_text = (candidates[0] + (" " + candidates[1] if len(candidates) > 1 else ""))
                        else:
                            top = [s for sc, s in scored[:2]]
                            response_text = " ".join(top)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    response_text = "I couldn't find that in the document."
            
            return {
                "result": response_text,
                "source_documents": docs
            }
    
    prompt_template = """
    Answer the question concisely based only on the following context. 
    If the answer isn't in the context, say "I couldn't find that in the document."
    Keep answers brief - maximum 2-3 sentences.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    return SimpleQAChain(vector_store, model_name, prompt_template)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="PDF Chat Assistant", 
        layout="centered",
        page_icon="📄"
    )
    
    # Allow user to paste an API key in the sidebar before setup
    st.title("📄 Chat with PDFs using Gemini")
    st.caption("Upload PDFs, ask questions, get instant answers")
    
    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.quota_warning = False
        st.session_state.model_name = "gemini-1.5-flash"
        st.session_state.api_key_input = ""
    
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
        st.markdown("---")
        st.subheader("API Key (optional)")
        st.write("If your Google API key is not set in environment or Streamlit secrets, paste it here for this session.")
        st.session_state.api_key_input = st.text_input("Google API Key (paste here)", value=st.session_state.get("api_key_input", ""), type="password")
        if st.button("Use this API Key") and st.session_state.api_key_input:
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key_input
            st.success("API key set for this session.\nProceed to Process PDFs.")
        st.markdown("---")
        st.subheader("Embedding Model (optional)")
        default_embed = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/textembedding-gecko-001")
        st.session_state.embedding_model_input = st.text_input(
            "Embedding model name (examples: models/textembedding-gecko-001)",
            value=st.session_state.get("embedding_model_input", default_embed)
        )
        st.caption("If empty, the app will try common defaults. Use the ListModels API or your Google AI console to find available embedding models.")
        st.markdown("[List available models — Google Generative AI](https://developers.generativeai.google/apis/models)")
        
        process_button = st.button("Process PDFs", disabled=not pdf_files)
        if process_button and pdf_files:
            # Ensure API key is available before processing
            try:
                api_key = setup_environment()
            except Exception:
                api_key = None
            if not api_key:
                st.error("Google API Key not found. Please set it in Streamlit secrets, environment, or paste it in the sidebar.")
                st.stop()
            with st.status("Processing documents..."):
                # Extract and process text
                st.write("📖 Reading PDF content...")
                raw_text = extract_pdf_text(pdf_files)
                
                st.write("✂️ Splitting text into chunks...")
                text_chunks = chunk_text(raw_text)
                
                st.write("🧠 Creating knowledge base...")
                preferred_model = st.session_state.get("embedding_model_input") or None
                st.session_state.vector_store = create_vector_store(text_chunks, preferred_model=preferred_model)
                
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
    if st.session_state.processed and st.session_state.vector_store:
        if not st.session_state.qa_chain:
            try:
                st.session_state.qa_chain = setup_qa_chain(
                    st.session_state.vector_store,
                    st.session_state.model_name
                )
            except Exception as e:
                st.error(f"⚠️ Error initializing AI system: {str(e)}")
                st.error("Please ensure you're using the latest libraries")
                st.stop()
        
        user_query = st.chat_input("Ask about your PDFs...")
        if user_query:
            st.chat_message("user").write(user_query)
            
            with st.spinner("🔍 Searching documents..."):
                try:
                    start_time = time.time()
                    
                    # Call the custom QA chain
                    response = st.session_state.qa_chain({"query": user_query})
                    
                    processing_time = time.time() - start_time

                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(response.get("result", "No answer generated"))
                        st.caption(f"⏱️ Response time: {processing_time:.1f} seconds | Model: {st.session_state.model_name}")

                        # Show source pages
                        if response.get("source_documents"):
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
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# import os
# import time
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize environment
# def setup_environment():
#     # Check Streamlit secrets first
#     if "GOOGLE_API_KEY" in st.secrets:
#         os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
#         return st.secrets["GOOGLE_API_KEY"]
    
#     # Then check environment variables
#     if api_key := os.getenv("GOOGLE_API_KEY"):
#         return api_key
    
#     # Finally try .env file
#     try:
#         from dotenv import load_dotenv
#         load_dotenv()
#         if api_key := os.getenv("GOOGLE_API_KEY"):
#             return api_key
#     except:
#         pass
    
#     st.error("Google API Key not found! Add it to Streamlit secrets.")
#     st.stop()

# # PDF Processing Functions
# def extract_pdf_text(pdf_files):
#     """Extract text from multiple PDFs with page numbers"""
#     text = ""
#     for pdf_file in pdf_files:
#         pdf_reader = PdfReader(pdf_file)
#         for page_num, page in enumerate(pdf_reader.pages):
#             if page_text := page.extract_text():
#                 text += f"--- Page {page_num+1} ---\n{page_text}\n\n"
#     return text

# def chunk_text(text, chunk_size=800, chunk_overlap=150):
#     """Split text into manageable chunks with optimized size"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ".", " ", ""]
#     )
#     return splitter.split_text(text)

# def create_vector_store(text_chunks):
#     """Create FAISS vector store in memory"""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return FAISS.from_texts(text_chunks, embedding=embeddings)

# # QA System Setup
# def setup_qa_chain(vector_store, model_name="gemini-1.5-flash"):
#     """Create retrieval-based QA system"""
#     prompt_template = """
#     Answer the question concisely based only on the following context. 
#     If the answer isn't in the context, say "I couldn't find that in the document."
#     Keep answers brief - maximum 2-3 sentences.
    
#     Context: {context}
    
#     Question: {question}
    
#     Answer:
#     """
    
#     # Use the selected model
#     model = ChatGoogleGenerativeAI(
#         model=model_name,
#         temperature=0.2,
#         max_output_tokens=300
#     )
    
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )
    
#     return RetrievalQA.from_chain_type(
#         llm=model,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
#         chain_type_kwargs={"prompt": prompt},
#         return_source_documents=True
#     )

# # Streamlit UI
# def main():
#     api_key = setup_environment()
    
#     st.set_page_config(
#         page_title="PDF Chat Assistant", 
#         layout="centered",
#         page_icon="📄"
#     )
#     st.title("📄 Chat with PDFs using Gemini")
#     st.caption("Upload PDFs, ask questions, get instant answers")
    
#     # Initialize session state
#     if "processed" not in st.session_state:
#         st.session_state.processed = False
#         st.session_state.vector_store = None
#         st.session_state.qa_chain = None
#         st.session_state.quota_warning = False
#         st.session_state.model_name = "gemini-1.5-flash"
    
#     # Sidebar for configuration
#     with st.sidebar:
#         st.subheader("Configuration")
        
#         # Model selection
#         model_option = st.selectbox(
#             "Select Gemini Model",
#             options=[
#                 ("🚀 Flash (Fast & Efficient)", "gemini-1.5-flash"),
#                 ("⚖️ Pro (Balanced)", "gemini-1.0-pro"),
#                 ("🧠 Advanced (Long Context)", "gemini-1.5-pro-latest")
#             ],
#             format_func=lambda x: x[0],
#             index=0
#         )
#         st.session_state.model_name = model_option[1]
        
#         st.subheader("Upload PDFs")
#         pdf_files = st.file_uploader(
#             "Select PDF documents", 
#             type="pdf", 
#             accept_multiple_files=True,
#             help="Upload one or more PDF files to analyze"
#         )
        
#         process_button = st.button("Process PDFs", disabled=not pdf_files)
#         if process_button and pdf_files:
#             with st.status("Processing documents..."):
#                 # Extract and process text
#                 st.write("📖 Reading PDF content...")
#                 raw_text = extract_pdf_text(pdf_files)
                
#                 st.write("✂️ Splitting text into chunks...")
#                 text_chunks = chunk_text(raw_text)
                
#                 st.write("🧠 Creating knowledge base...")
#                 st.session_state.vector_store = create_vector_store(text_chunks)
                
#                 st.session_state.processed = True
#                 st.session_state.quota_warning = False
#                 st.success("✅ PDFs processed successfully! You can now ask questions.")
                
#         # Reset button
#         if st.button("🔄 Reset Session"):
#             st.session_state.clear()
#             st.rerun()
                
#         # Quota information
#         st.markdown("---")
#         st.info("**Free Tier Limitations:**\n"
#                 "- 60 requests/minute\n"
#                 "- 1,500 requests/day\n\n"
#                 "Upgrade at [Google AI Studio](https://aistudio.google.com/)")

#     # Main chat interface
#     if st.session_state.processed and st.session_state.vector_store:
#         if not st.session_state.qa_chain:
#             try:
#                 st.session_state.qa_chain = setup_qa_chain(
#                     st.session_state.vector_store,
#                     st.session_state.model_name
#                 )
#             except Exception as e:
#                 st.error(f"⚠️ Error initializing AI system: {str(e)}")
#                 st.error("Please ensure you're using the latest libraries")
#                 st.stop()
        
#         user_query = st.chat_input("Ask about your PDFs...")
#         if user_query:
#             st.chat_message("user").write(user_query)
            
#             with st.spinner("🔍 Searching documents..."):
#                 try:
#                     start_time = time.time()
#                     response = st.session_state.qa_chain({"query": user_query})
#                     processing_time = time.time() - start_time
                    
#                     with st.chat_message("assistant", avatar="🤖"):
#                         st.write(response["result"])
#                         st.caption(f"⏱️ Response time: {processing_time:.1f} seconds | Model: {st.session_state.model_name}")
                        
#                         # Show source pages
#                         if response["source_documents"]:
#                             with st.expander("🔍 Source Information"):
#                                 for i, doc in enumerate(response["source_documents"]):
#                                     page_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
#                                     st.caption(f"**Source {i+1}**")
#                                     st.info(page_content)
#                 except Exception as e:
#                     if "quota" in str(e).lower() or "429" in str(e):
#                         st.error("⚠️ API Quota Exceeded - You've hit the free tier limits")
#                         st.error("Please try again later or upgrade your Google Cloud account")
#                         st.session_state.quota_warning = True
#                     else:
#                         st.error(f"❌ Error processing your question: {str(e)}")
    
#     # Quota warning display
#     if st.session_state.get("quota_warning", False):
#         st.warning("📢 You've exceeded your free tier quota. Here are your options:")
#         st.markdown("""
#         1. **⏳ Wait 1 minute** - Free tier resets every minute
#         2. **💳 Upgrade your account** - [Google AI Studio Pricing](https://ai.google.dev/pricing)
#         3. **🔑 Use a different API key** - If you have multiple projects
#         4. **📉 Reduce usage** - Ask fewer questions or switch to Flash model
#         """)
    
#     # Initial instructions
#     if not st.session_state.processed:
#         st.info("👋 Welcome! To get started:")
#         st.markdown("""
#         1. 👉 **Upload PDF files** in the sidebar
#         2. 🚀 Click **Process PDFs** to build the knowledge base
#         3. 💬 Start asking questions in the chat
        
#         For best results:
#         - Use the Flash model for quick responses
#         - Keep questions focused on document content
#         - Process only necessary documents
#         """)
#         st.image("https://images.unsplash.com/photo-1497636577773-f1231844b336?auto=format&fit=crop&w=600", 
#                  caption="AI Document Analysis")

# if __name__ == "__main__":
#     main()
