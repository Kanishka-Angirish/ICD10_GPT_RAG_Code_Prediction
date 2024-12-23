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
import torch

# Pre-trained BioGpt Model
causal_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")

# Pre-trained tokenizer for Bio Gpt
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large", clean_up_tokenization_spaces=True)
causal_model.to("mps")

# hUGGING FACE PIPELINE FOR BIOGPT WHICH COULD BE USED FOR ANSWERING USER QUERY
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