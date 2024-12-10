"""
This file consist of the BioGPT Models used for different purposes in the project

Author: Kanishka Angirish
"""

from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForTokenClassification
from transformers import BioGptTokenizer, BioGptForCausalLM, BioGptForTokenClassification, TokenClassificationPipeline
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.llms import HuggingFacePipeline

# Pre-trained BioGpt Model
causal_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")
classification_model = AutoModelForTokenClassification.from_pretrained("microsoft/biogpt-large")

# Pre-trained tokenizer for Bio Gpt
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large", clean_up_tokenization_spaces=True)
causal_model.to("mps")

# hUGGING FACE PIPELINE FOR BIOGPT WHICH COULD BE USED FOR ANSWERING USER QUERY
causal_generator = pipeline("text-generation", model=causal_model, tokenizer=tokenizer)
ner_pipeline = TokenClassificationPipeline(model=classification_model, tokenizer=tokenizer,
                                           aggregation_strategy="simple")

# Define the tools for NER Classification model
def ner_tool(text: str) -> str:
    entities = ner_pipeline(text)
    output = "\n".join([f"{entity}" for entity in entities])
    return output

ner_langchain_tool = Tool(
    name = "Named Entity Recognition",
    func = ner_tool,
    description = "Extracts named entity from biomedical texts and entity"
)

# Wrap the hugging face pipeline around LangChain
causal_llm = HuggingFacePipeline(pipeline=causal_generator)


# classification_llm = HuggingFacePipeline(pipeline=ner_pipeline)

ner_agent  = initialize_agent(
    tools=[ner_langchain_tool],
    agent = "zero-shot-react-description",
    llm= HuggingFacePipeline(pipeline=ner_pipeline)
)


# Code for testing

prompt = PromptTemplate(template="{caseinfo}",
                        input_variables=["caseinfo"])
chain = LLMChain(llm=causal_llm, prompt=prompt)
response = chain.invoke("COVID-19 is")
print(f"Text Completion Response is: {response}")

ner_prompt = PromptTemplate(template="Extract the named entity from the following biomedical text: \n\n {text}",
                            input_variables=["text"])
result = ner_agent.run("Extract named entities from the text: The patient has diabetes mellitus and takes metformin.")
print(f"NER Response is: {result}")
