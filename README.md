# ICD-10 Code Prediction Repository

## Overview

This repository contains a Python-based application for predicting ICD-10 codes using advanced large language model such as Azure OpenAI GPT-4 32K. The application leverages embeddings, FAISS vector store, and prompt engineering to analyze user-provided case descriptions and generate the most appropriate ICD-10 codes. It also supports knowledgebase updates for biomedical data, code dictionaries, and patient reports.

### Key Features:
- **ICD-10 Code Prediction**: Predict ICD-10 codes based on case descriptions or patient reports.
- **Knowledgebase Update**: Add new biomedical data, code dictionaries, or patient reports to the vector store.
- **Integration with Azure OpenAI**: Utilize Azure OpenAI models for embeddings and predictions.
- **RAG (Retrieval-Augmented Generation)**: Combine retrieval of relevant data with generation of responses for improved accuracy.
- **Prompt Engineering**: Use structured prompts to enhance model performance.
---

## Setup Instructions

### Prerequisites
1. **Python**: Ensure Python 3.9 or higher is installed.
2. **Virtual Environment**: Recommended for dependency management.
3. **API Keys**: Obtain OpenAI and Azure OpenAI API keys.

---

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kanishka-angirish/discharge_summary_data_gpt.git
   cd discharge_summary_data_gpt

2. **Create a Virtual Environment**:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ``` 
4. **Set Environment Variables:** Update the `.env` file with your API keys and endpoint:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   ```
<hr></hr>

## Running the Application

**Modes of Operation**:

The application supports two modes:  

1. Knowledgebase Update: Add new data to the vector store.
2. ICD Code Prediction: Predict ICD-10 codes based on case descriptions or reports.

**Commands**

1. **Run the Application:**
   ```bash
   python src/main.py
   ```
2. **Choose Mode**: 
   - Enter `1` for Knowledgebase Update.
     - Follow the prompts to add case descriptions, code dictionaries, or patient reports.
     
   - Enter `2` for ICD Code Prediction.
        - Input the case description or patient report.
        - The application will return the predicted ICD-10 codes.
<hr> </hr>

## File Structure

- `src/main.py`: Entry point for the application.
- `src/bioGPT.py`: Contains BioGPT model setup and integration.
- `src/prompts.py`: Defines prompt templates for ICD-10 code prediction.
- `src/utils.py`: Utility functions for logging, vector store management, and data processing.
- `src/faiss_vector_store.py`: Handles vector store operations using FAISS.
- `.env`: Stores environment variables for API keys and endpoints.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<hr></hr>

## Author
Kanishka Angirish

