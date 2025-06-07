# DOCUCHATS - PDF Upload and Question Answering System

This project allows users to upload PDF documents, processes their content, and answers questions based on the PDF data using an AI language model.

---

## Features

- Upload PDF files via a web interface (built with Streamlit).
- Extract text content from uploaded PDFs.
- Split large documents into manageable chunks for efficient processing.
- Create vector embeddings of text chunks for fast similarity search.
- Store embeddings **in-memory** for quick access during the session.
- Use a Language Model (LLM) to generate accurate answers based on the PDF content.
- Interactive Q&A with users referencing uploaded PDFs.

---

## How It Works

### Workflow

1. **User Uploads PDF:**  
   The user uploads one or more PDF documents through the Streamlit interface.

2. **Text Extraction:**  
   The system extracts text from each PDF.

3. **Text Splitting:**  
   The extracted text is split into smaller chunks to allow more efficient embedding and querying.

4. **Embedding Generation:**  
   Each chunk is converted into vector embeddings using an embedding model (e.g., OpenAI embeddings).

5. **Indexing:**  
   These embeddings are stored in an **in-memory vector store** for fast similarity search.

6. **User Question:**  
   The user asks a question related to the PDF content.

7. **Similarity Search:**  
   The system retrieves relevant chunks by searching the vector store.

8. **Answer Generation:**  
   The LLM generates an answer based on the retrieved chunks.

9. **Display Answer:**  
   The answer is displayed back to the user in the Streamlit interface.

---

## Storage Details

- **PDF Files:** Temporarily stored in the server’s temporary folder (or in memory if handled via Streamlit file uploader).  
- **Text and Embeddings:** Stored in memory during the app session — typically RAM on the deployment server where your app runs (not on the user’s local RAM).  
- **Persistence:** No persistent database is used in this version. When the app restarts, all data is cleared.

---

## Tech Stack

- Python 3.10+  
- Streamlit (Web UI)  
- PyPDF2 (PDF text extraction)  
- LangChain (text splitting, embeddings, vectorstore)  
- OpenAI / Google Generative AI embeddings  
- FAISS or in-memory vector store  

---
## Diagrams
![image](https://github.com/user-attachments/assets/c9e90fbe-30ac-4210-8e49-73ba32530156)
![image](https://github.com/user-attachments/assets/4deae04a-3ccf-4382-aec6-efb48bfb6ad0)



## Installation

```bash
pip install -r requirements.txt
streamlit run app.py




