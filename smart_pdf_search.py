import streamlit as st
import os
from datetime import datetime
import PyPDF2
import sqlite3
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import base64
import plotly.express as px
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="📚 Smart Document Manager",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5em 1em;
        border-radius: 5px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

class DocumentManagementSystem:
    def __init__(self, storage_path: str = "document_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.db_path = self.storage_path / "documents.db"
        self.init_database()
        
        # Initialize models with @st.cache_resource for better performance
        self.load_models()

    @st.cache_resource
    def load_models(_self):
        """Load ML models with caching"""
        _self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        _self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        _self.model = AutoModel.from_pretrained("bert-base-uncased")

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP,
                content TEXT,
                embedding BLOB,
                file_size INTEGER,
                page_count INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def extract_text_from_pdf(self, file_path: Path) -> tuple:
        """Extract text content and metadata from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        file_size = os.path.getsize(file_path)
        return text, file_size, page_count

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.embeddings_model.encode(text)
    class DocumentManagementSystem:

        def __init__(self):

            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def embed_text(self, text):
        return self.embedding_model.encode(text)


    def store_document(self, file) -> dict:
        """Store uploaded document and metadata"""
        file_path = self.storage_path / file.name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)

        text_content, file_size, page_count = self.extract_text_from_pdf(file_path)
        embedding = self.generate_embedding(text_content)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (filename, upload_date, content, embedding, file_size, page_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file.name, datetime.now(), text_content, embedding.tobytes(), file_size, page_count))
        
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {
            "id": doc_id,
            "filename": file.name,
            "file_size": file_size,
            "page_count": page_count
        }

    def search_documents(self, query: str, top_k: int = 5) -> list:
        """Search documents using semantic similarity"""
        query_embedding = self.generate_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filename, content, embedding, file_size, page_count, upload_date 
            FROM documents
        """)
        results = cursor.fetchall()
        conn.close()

        similarities = []
        for doc_id, filename, content, emb_bytes, size, pages, date in results:
            doc_embedding = np.frombuffer(emb_bytes, dtype=np.float32)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
            similarities.append({
                "id": doc_id,
                "filename": filename,
                "content": content[:200] + "...",
                "similarity": float(similarity),
                "file_size": size,
                "page_count": pages,
                "upload_date": date
            })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def get_document_stats(self):
        """Get statistics about stored documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as doc_count, 
                   SUM(page_count) as total_pages,
                   SUM(file_size) as total_size
            FROM documents
        """)
        stats = cursor.fetchone()
        
        cursor.execute("SELECT filename, page_count, upload_date FROM documents")
        doc_details = cursor.fetchall()
        conn.close()
        
        return {
            "doc_count": stats[0] or 0,
            "total_pages": stats[1] or 0,
            "total_size": stats[2] or 0,
            "doc_details": doc_details
        }

    def answer_question(self, question: str, doc_id: int = None) -> str:
        """Generate answer to question based on document context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if doc_id:
            cursor.execute("SELECT content FROM documents WHERE id = ?", (doc_id,))
        else:
            cursor.execute("SELECT content FROM documents")
            
        contents = cursor.fetchall()
        conn.close()

        if not contents:
            return "No documents found to answer the question."

        context = " ".join([content[0] for content in contents])
        sentences = context.split('.')
        sentence_embeddings = self.embeddings_model.encode(sentences)
        question_embedding = self.embeddings_model.encode(question)
        
        similarities = cosine_similarity(
            question_embedding.reshape(1, -1),
            sentence_embeddings
        )[0]
        
        most_relevant_idx = np.argmax(similarities)
        return sentences[most_relevant_idx].strip()

def main():
    st.title("📚 Smart Document Management System")
    st.markdown("### Your AI-Powered Document Assistant 🤖")

    # Initialize DMS
    dms = DocumentManagementSystem()

    # Sidebar
    st.sidebar.title("Navigation 🧭")
    page = st.sidebar.radio("Choose a page:", 
        ["📊 Dashboard", "📤 Upload Documents", "🔍 Search & Query", "❓ Q&A Assistant"])

    if page == "📊 Dashboard":
        st.header("📊 System Dashboard")
        stats = dms.get_document_stats()
        
        # Display key metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📑 Total Documents", stats["doc_count"])
        with col2:
            st.metric("📄 Total Pages", stats["total_pages"])
        with col3:
            st.metric("💾 Total Size (MB)", f"{stats['total_size']/1024/1024:.2f}")

        # Document timeline
        if stats["doc_details"]:
            df = pd.DataFrame(stats["doc_details"], 
                            columns=["filename", "pages", "upload_date"])
            df["upload_date"] = pd.to_datetime(df["upload_date"])
            
            fig = px.line(df, x="upload_date", y="pages", 
                         title="📈 Document Upload Timeline",
                         labels={"upload_date": "Upload Date", "pages": "Number of Pages"})
            st.plotly_chart(fig)

    elif page == "📤 Upload Documents":
        st.header("📤 Upload New Documents")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if uploaded_file:
            with st.spinner("📥 Processing document..."):
                try:
                    result = dms.store_document(uploaded_file)
                    st.success(f"✅ Successfully uploaded: {result['filename']}")
                    st.info(f"""
                        📊 Document Statistics:
                        - Pages: {result['page_count']}
                        - Size: {result['file_size']/1024:.1f} KB
                    """)
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

    elif page == "🔍 Search & Query":
        st.header("🔍 Search Documents")
        search_query = st.text_input("Enter your search query:")
        
        if search_query:
            with st.spinner("🔎 Searching..."):
                results = dms.search_documents(search_query)
                
                if results:
                    st.success(f"Found {len(results)} relevant documents:")
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"📄 {idx}. {result['filename']} (Score: {result['similarity']:.2f})"):
                            st.markdown(f"""
                                - 📅 Upload Date: {result['upload_date']}
                                - 📄 Pages: {result['page_count']}
                                - 💾 Size: {result['file_size']/1024:.1f} KB
                                - 📝 Preview: {result['content']}
                            """)
                else:
                    st.warning("No matching documents found.")

    elif page == "❓ Q&A Assistant":
        st.header("❓ Document Q&A Assistant")
        question = st.text_input("Ask a question about your documents:")
        
        if question:
            with st.spinner("🤔 Thinking..."):
                answer = dms.answer_question(question)
                st.info(f"💡 Answer: {answer}")

if __name__ == "__main__":
    main()