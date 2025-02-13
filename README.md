# Smart PDF Search

## Overview
Smart PDF Search is an AI-powered search engine that enables users to efficiently search through PDF documents using natural language queries. This tool utilizes advanced NLP techniques to extract and index text from PDFs, providing quick and relevant search results.

## Features
- **Natural Language Processing (NLP):** Enables intelligent querying using conversational language.
- **OCR Support:** Extracts text from scanned PDFs using Optical Character Recognition.
- **Fast and Efficient Search:** Uses indexing techniques for quick retrieval of relevant sections.
- **User-Friendly Interface:** Provides a simple UI for uploading PDFs and conducting searches.
- **Multi-PDF Search:** Supports searching across multiple documents simultaneously.

## Technologies Used
- Python
- Streamlit (for UI)
- PyMuPDF / pdfplumber (for PDF text extraction)
- OCR with Tesseract (for scanned PDFs)
- FAISS / Elasticsearch (for indexing and fast retrieval)
- Hugging Face Transformers / OpenAI API (for NLP-based querying)

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/smart-pdf-search.git
   cd smart-pdf-search
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload one or multiple PDF documents.
2. Enter a query in natural language.
3. View highlighted and extracted relevant sections.

## Future Enhancements
- Support for audio-based search.
- Integration with cloud storage (Google Drive, Dropbox, etc.).
- Advanced semantic search with knowledge graphs.

## Contributing
Contributions are welcome! Feel free to submit issues and pull requests to improve the project.

## License
This project is licensed under the MIT License.


