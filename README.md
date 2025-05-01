# LegalMind Assistant

## Overview

LegalMind Assistant is an LLM-based application designed to democratize access to legal information and reasoning. Unlike traditional legal research tools that primarily serve legal professionals, LegalMind aims to make legal information accessible to everyone, from lawyers to ordinary citizens seeking to understand their rights and responsibilities under Australian law.

## Motivation

Legal information remains largely inaccessible to the general public due to:

- Specialized legal terminology and jargon
- Complex legal documents and precedents
- Prohibitive costs of traditional legal services
- Difficulty in interpreting how laws apply to specific situations

LegalMind addresses these challenges by providing an intuitive interface for querying legal information, understanding precedents, and receiving plain-language explanations of complex legal concepts.

## Features

- **Natural Language Querying**: Ask legal questions in plain language
- **Australian Legal Context**: Specialized in Australian law and legal precedents
- **Citation-Aware Responses**: Provides proper legal citations
- **Plain Language Explanations**: Makes complex legal concepts accessible
- **Jurisdiction Filtering**: Filter results by Australian state/territory
- **Document Type Filtering**: Focus on specific kinds of legal documents

## Technical Architecture

LegalMind is built using the following components:

### 1. Data Pipeline

- **Dataset**: Uses Australian legal Q&A datasets from Hugging Face
- **Specialized Legal Document Chunking**:
    - Top-level chunking by case document
    - Mid-level chunking by legal reasoning sections
    - Paragraph-level chunking for specific legal points
- **Metadata Enrichment**: Preserves jurisdiction data, document type, citations

### 2. Embedding and Vector Storage

- **Embedding Model**: BGE-large-en-v1.5
- **Vector Database**: Chroma

### 3. Retrieval System

- **Basic RAG**: Retrieves relevant legal documents for queries
- **Metadata-Enhanced Filtering**: Filters by jurisdiction, document type, case year

### 4. Language Model

- **LLM**: Mistral-7B-Instruct-v0.2 / Phi4-mini
- **Legal Context Augmentation**: Enhances responses with retrieved legal context
- **Citation Recognition**: Identifies and preserves legal citations

### 5. Evaluation Framework

- **Retrieval Metrics**: Precision, recall, semantic similarity
- **Response Quality**: Citation accuracy, completeness, hallucination detection

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/legalmind.git
cd legalmind
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download and process the dataset:
```
python scripts/download_dataset.py
python scripts/process_data.py
```

## Usage

1. Start the Streamlit UI:
```
streamlit run src/ui/app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

3. Ask legal questions and explore the retrieved documents.

## Project Structure

```
legalmind/
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── config/
│   └── config.yaml                # Configuration parameters
├── data/
│   ├── raw/                       # Raw legal datasets
│   └── processed/                 # Processed chunks
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Data loading and processing
│   │   ├── chunking.py            # Legal document chunking strategies
│   │   └── preprocessing.py       # Text cleaning and preprocessing
│   ├── embeddings/
│   │   └── embedding.py           # BGE embedding implementation
│   ├── vectordb/
│   │   └── chroma_db.py           # Chroma vector database utilities
│   ├── retrieval/
│   │   ├── basic_rag.py           # Basic RAG implementation
│   │   └── metadata_filter.py     # Metadata filtering utilities
│   ├── models/
│   │   └── llm.py                 # LLM implementation (Mistral/Phi4)
│   ├── evaluation/
│   │   └── metrics.py             # Evaluation metrics
│   └── ui/
│       └── app.py                 # Streamlit UI
└── scripts/
    ├── download_dataset.py        # Script to download datasets
    ├── process_data.py            # Data processing pipeline
    └── run_evaluation.py          # Evaluation script
```

## Future Improvements

- **Advanced RAG Strategies**: Query expansion with legal terminology, multi-query approaches
- **Fine-tuning**: Task-specific fine-tuning for legal Q&A
- **RLHF**: Lightweight RLHF approach for legal accuracy
- **Hallucination Mitigation**: Citation verification, confidence scoring
- **Jurisdiction-Specific Models**: Specialized models for different Australian jurisdictions

## Disclaimer

LegalMind provides legal information, not legal advice. The information provided is intended for educational purposes only and should not be construed as legal advice. For legal advice specific to your situation, please consult a qualified legal professional.

## License

This project is licensed under the MIT License - see the LICENSE file for details.