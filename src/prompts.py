from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

inference_prompt_template = """
You are a helpful biomedical assistant and your task is to interpret the case and infer the valuable information.
Refer to the below example to understand the task. DO NOT USE any information from the example while inferring the valuable information
from the case description provided by user. USE IT ONLY FOR UNDERSTANDING THE TASK.

For example:
============
## Example Case Description: Patient is a 65-year-old female presenting with symptoms of heart failure. 
Echocardiogram confirms nonrheumatic aortic valve insufficiency, with a regurgitant fraction of 30%. 
Patient reports fatigue and mild peripheral edema. Surgical valve replacement is being considered.

## Valuable Information: Nonrheumatic aortic (valve) insufficiency

Provide the inferred valuable information as an output in the following json format:

"inferred_info": "valuable_information"
"""

icd_code_prediction_prompt_template = """
You are a helpful biomedical assistant and your task is to predict the ICD-10 code for the given case description.
Also refer to the following example to understand how ICD-10 code is generated for respective case description and 
corresponding valuable information. DO NOT USE any information from the example while predicting the ICD-10 code for the 
user provided case description. USE IT ONLY FOR UNDERSTANDING THE TASK.

## Examples for Understanding the Task
=======================================

{context}

## User provided case description 
==================================
 
Case Description: {caseinfo}
Valuable Information: {inferred_info} 
ICD-10 Code is:
"""
inference_prompt = PromptTemplate(template=inference_prompt_template, input_variables=["caseinfo"])
inference_chat_prompt = ChatPromptTemplate.from_messages(messages=[
    (
        "system",
        inference_prompt_template
    ),
    ("human", "{caseinfo}")
])

prediction_chat_prompt = ChatPromptTemplate.from_messages(messages=[
    (
        "system",
        icd_code_prediction_prompt_template
    ),
    ("human", "{caseinfo} \n {inferred_info} \n {context}")
])
prediction_prompt = PromptTemplate(template=icd_code_prediction_prompt_template, input_variables=["caseinfo",
                                                                                                  "inferred_info",
                                                                                                  "context"])
