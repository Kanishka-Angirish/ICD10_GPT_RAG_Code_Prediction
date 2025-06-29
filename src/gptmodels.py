"""
This file consist of the BioGPT Models used for different purposes in the project

Author: Kanishka Angirish
"""

from transformers import pipeline
from transformers import (
    BioGptTokenizer, BioGptForCausalLM
)
from langchain.llms import HuggingFacePipeline
from langchain_openai import AzureOpenAIEmbeddings, AzureOpenAI, AzureChatOpenAI
import os

from utils import OPENAI_API_KEY, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

# Setting up API KEY AND ENDPOINT
os.environ["OPENAI_API_VERSION"] = "2024-10-21"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY

# Pre-trained BioGpt Model
causal_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")

# Azure OpenAI Model
llm_model = AzureOpenAI(
    azure_deployment="gpt-35-turbo-instruct",
    api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0,
    max_retries=2
)

# Azure Embedding Model
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

# Azure GPT-4 Model
chat_model = AzureChatOpenAI(
    azure_deployment="gpt-4-32k",
    openai_api_version="2024-10-21",
    temperature=0,
    max_retries=2
)

# ============ Deprecated Code ============
# Pre-trained tokenizer for Bio Gpt
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large", clean_up_tokenization_spaces=True)
# causal_model.to("mps")

# HUGGING FACE PIPELINE FOR BIOGPT WHICH COULD BE USED FOR ANSWERING USER QUERY
# causal_generator = pipeline("text-generation", model=causal_model, tokenizer=tokenizer)

# Wrap the hugging face pipeline around LangChain
# causal_llm = HuggingFacePipeline(pipeline=causal_generator)
