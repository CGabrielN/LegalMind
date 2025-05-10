# LegalMind Assistant

## Overview

LegalMind Assistant is an advanced LLM-based application designed to democratize access to legal information and reasoning, with a focus on Australian law. It goes beyond traditional legal research tools by implementing state-of-the-art RAG (Retrieval Augmented Generation) strategies, hallucination mitigation, and RLHF (Reinforcement Learning from Human Feedback) to provide accurate, reliable legal information to everyone, from legal professionals to ordinary citizens.

## Motivation

Legal information remains largely inaccessible to the general public due to:

- Specialized legal terminology and jargon
- Complex legal documents and precedents
- Prohibitive costs of traditional legal services
- Difficulty in interpreting how laws apply to specific situations

LegalMind addresses these challenges by providing an intuitive interface for querying legal information, understanding precedents, and receiving plain-language explanations of complex legal concepts.

## Key Features

- **Multiple Advanced RAG Strategies** - Several retrieval approaches tailored to legal queries:
  - **Basic RAG** - Standard vector similarity retrieval
  - **Query Expansion with Legal Terminology** - Enhances queries with legal synonyms and jurisdiction-specific terms
  - **Multi-Query for Legal Perspectives** - Captures different viewpoints (plaintiff, defendant, statutory, etc.)
  - **Metadata-Enhanced Retrieval** - Leverages jurisdiction, court hierarchy, recency, and citation networks
  - **Automatic Strategy Selection** - Intelligently chooses the best retrieval approach for each query type

- **Hallucination Mitigation** - Ensures factual accuracy in legal responses:
  - **Citation Verification** - Validates legal citations against context
  - **Legal Claim Detection** - Identifies definitive statements that require support
  - **Jurisdiction Consistency** - Ensures responses stay within the appropriate legal jurisdiction
  - **Response Improvement** - Automatically corrects or adds disclaimers to potential hallucinations

- **RLHF System** - Improves response quality through feedback:
  - **Preference Learning** - Trains on human preferences between response pairs
  - **Reward Modeling** - Scores responses based on legal quality and reliability
  - **Feedback Collection** - Gathers user preferences through the UI

- **Citation-Aware Responses** - Provides proper Australian legal citations with verification
- **Plain Language Explanations** - Makes complex legal concepts accessible
- **Jurisdiction-Specific Information** - Tailors responses to Australian legal jurisdictions

## Technical Architecture

### 1. Data Pipeline

- **Australian Legal Datasets** - Uses specialized legal Q&A collections from Hugging Face
- **Legal Document Chunking** - Implements domain-specific chunking strategies:
  - Top-level chunking by case document
  - Mid-level chunking by legal reasoning sections (facts, arguments, holdings)
  - Paragraph-level chunking for specific legal points
- **Metadata Enrichment** - Preserves jurisdiction, document type, citation information

### 2. Advanced RAG Implementation

#### Query Expansion with Legal Terminology
- Identifies legal terms in queries
- Expands with synonyms and related concepts
- Adds jurisdiction-specific terminology

#### Multi-Query RAG for Legal Perspectives
- Generates different perspectives (plaintiff, defendant)
- Creates jurisdiction-specific query variations
- Combines and deduplicates results from multiple queries

#### Metadata-Enhanced Retrieval
- **Jurisdiction Filtering** - Focuses on relevant Australian jurisdictions
- **Court Hierarchy Weighting** - Prioritizes higher courts (High Court > Supreme Court > etc.)
- **Recency Analysis** - Adjusts relevance based on case year (newer cases often supersede older ones)
- **Citation Network Mapping** - Uses citation relationships to identify authoritative cases

### 3. Embedding and Vector Storage

- **Embedding Model**: BGE-large-en-v1.5
- **Vector Database**: Chroma

### 4. Hallucination Mitigation System

- **Citation Handling**
  - Extraction and parsing of Australian legal citations
  - Verification against retrieved context
  - Formatting according to Australian citation style

- **Hallucination Detection**
  - Unverified citation detection
  - Definitive legal claim identification
  - Jurisdiction consistency checking
  - Confidence scoring for response reliability

- **Response Improvement**
  - Citation correction or disclaimer addition
  - Definitive language softening where appropriate
  - Jurisdiction-specific disclaimers
  - Confidence-based information presentation

### 5. RLHF Implementation

- **Preference Dataset** - Collection of preferred vs. rejected response pairs
- **Reward Model** - Fine-tunable model for scoring response quality
- **Feedback Collection** - UI components for gathering user preferences
- **Training Pipeline** - Continuous improvement from collected feedback

### 6. User Interface

- **Streamlit-based UI** with conversation memory
- **Strategy Selection** - Choose between different RAG approaches
- **Confidence Display** - Transparency about response reliability
- **Document Inspection** - View retrieved legal documents with relevance scores
- **Response Analysis** - View citation verification and hallucination detection results
- **Feedback Collection** - Compare and rate alternative responses

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
python -m src.main --download --process
```

4. Initialize the RLHF system:
```
python -m src.main --init-rlhf
```

## Usage

1. Start the Streamlit UI:
```
python -m src.main --ui
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

3. Ask legal questions and explore the retrieved documents and analysis.

## Running the Evaluation

To evaluate different RAG strategies, you can use the custom evaluation script. This script is designed to test the performance of various retrieval strategies on a set of legal queries.

```bash
# Run the full evaluation (includes generation and evaluation of responses)
python scripts/custom_evaluation.py --phase all

# Run only the generation phase
python scripts/custom_evaluation.py --phase generate

# Run only the evaluation phase on previously generated responses
python scripts/custom_evaluation.py --phase evaluate --responses-file [path_to_responses_file]

# Run only the analysis phase to get a summary of results
python scripts/custom_evaluation.py --phase analyze --responses-file [path_to_evaluated_responses_file]

# Additional options
python scripts/custom_evaluation.py --help
```

### Evaluation Parameters

- `--config`: Specify a custom configuration file path
- `--test-set`: Specify a JSON file containing test queries
- `--lm-studio-url`: URL for your LM Studio instance (default: http://127.0.0.1:1234/v1)
- `--strategies`: Space-separated list of strategies to evaluate (default: all strategies)
- `--num-queries`: Number of test queries to evaluate (default: 5)
- `--output-dir`: Directory to save evaluation results
- `--use-heuristics`: Use heuristic evaluation instead of LLM-based evaluation (default: True)

### Evaluation Metrics

The evaluation uses the following metrics:
1. Relevancy - How relevant the response is to the query
2. Citation quality - Presence and accuracy of legal citations
3. Structure - Organization and formatting of the response
4. Context usage - How well the response incorporates retrieved context
5. Hallucination - Inverse measure of factual inaccuracies (higher is better)

## Performance Findings

Based on our comprehensive evaluation using standard legal queries, we found the following performance results:

### RAG Strategy Performance (Overall Scores)

| Rank | Strategy | Average Score |
|------|----------|---------------|
| 1 | Multi-Query RAG | 0.629 |
| 2 | Metadata-Enhanced RAG | 0.588 |
| 3 | Advanced (Auto-Select) RAG | 0.568 |
| 4 | Basic RAG | 0.493 |
| 5 | Query Expansion RAG | 0.484 |

### Strategy Strengths by Query Type

1. **Multi-Query RAG** (0.629) - Excels at complex legal questions with multiple perspectives, performing exceptionally well on jurisdiction-specific queries such as "How does adverse possession work in Victoria?" (0.832) and "What rights do tenants have under Queensland rental laws?" (0.800).

2. **Metadata-Enhanced RAG** (0.588) - Shows significant improvement for jurisdiction-specific queries and detailed legal concepts. Particularly strong in "How are damages calculated in breach of contract cases?" (0.900).

3. **Advanced (Auto-Select) RAG** (0.568) - Provides balanced performance across different query types, with consistent scores around 0.530-0.663. While not outperforming specialized strategies in their areas of strength, it delivers reliable results across diverse legal topics.

4. **Basic RAG** (0.493) - Performs adequately for simple queries but struggles with complex or jurisdiction-specific questions. Shows strength in specific cases with highly relevant documents (0.900 on tenant rights query).

5. **Query Expansion RAG** (0.484) - Generally underperforms compared to other strategies, though it shows improvement over Basic RAG for certain legal concept explanations.

### Query Performance Analysis

The evaluation tested a diverse set of legal queries:
- General legal principles ("What constitutes negligence in New South Wales?")
- Legal concept explanations ("Explain the duty of care concept in Australian tort law")
- Jurisdiction-specific questions ("How does adverse possession work in Victoria?")
- Procedural questions ("What is the process for appealing a court decision to the High Court of Australia?")
- Quantitative legal aspects ("How are damages calculated in breach of contract cases?")

Performance varied significantly across query types, with jurisdiction-specific and multi-perspective queries benefiting most from advanced RAG strategies.

### Hallucination Mitigation Impact

Our evaluation indicates that advanced RAG strategies generally produce fewer hallucinations:
- Multi-Query RAG showed a 33% reduction in hallucinations compared to Basic RAG
- Metadata-Enhanced RAG reduced hallucinations by approximately 28%
- The Advanced (Auto-Select) strategy provided a consistent reduction of about 25%

## System Limitations

Current limitations include:

1. Dependency on the quality of the retrieved context
2. Australian legal focus with limited coverage of other jurisdictions
3. Need for ongoing feedback to continuously improve the RLHF system
4. Processing time increases with more advanced RAG strategies
5. Limited to the capabilities of the underlying LLM (through LM Studio API)

## Legal Disclaimer

LegalMind provides legal information, not legal advice. The information provided is intended for educational purposes only and should not be construed as legal advice. For legal advice specific to your situation, please consult a qualified legal professional.

## License

This project is licensed under the MIT License - see the LICENSE file for details.