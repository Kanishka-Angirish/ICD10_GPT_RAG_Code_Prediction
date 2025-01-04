"""
This file consist of the BioGPT Models used for different purposes in the project

Author: Kanishka Angirish
"""
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from transformers import (
    BioGptTokenizer, BioGptForCausalLM
)
from langchain.llms import HuggingFacePipeline
from langchain_openai import AzureOpenAI
import torch
import os
from utils import OPENAI_API_KEY, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

# Setting up API KEY AND ENDPOINT
os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY

# Pre-trained BioGpt Model
causal_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")
llm_model = AzureOpenAI(
    azure_deployment="gpt-35-turbo-instruct",
    api_version = os.environ["OPENAI_API_VERSION"],
    temperature=0,
    max_retries=2
)

# Pre-trained tokenizer for Bio Gpt
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large", clean_up_tokenization_spaces=True)
causal_model.to("mps")

# HUGGING FACE PIPELINE FOR BIOGPT WHICH COULD BE USED FOR ANSWERING USER QUERY
causal_generator = pipeline("text-generation", model=causal_model, tokenizer=tokenizer)

# Wrap the hugging face pipeline around LangChain
causal_llm = HuggingFacePipeline(pipeline=causal_generator)

"""
prompt = PromptTemplate(template="{caseinfo}",
                        input_variables=["caseinfo"])
chain = LLMChain(llm=causal_llm, prompt=prompt)
response = chain.invoke("Consider yourself a patient with a history of heart disease and diabetes. You are experiencing chest pain and shortness of breath. What should you do?")
print(f"Text Completion Response is: {response}")
"""