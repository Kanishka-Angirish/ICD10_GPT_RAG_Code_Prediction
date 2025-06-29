from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

gpt_diagnosis_inference_prompt_template = """
You are a skilled ICD-10 medical coder who identifies medical diagnoses from clinical documentation. 
Your task is to analyze the provided text and extract all medical diagnoses, including explicit and implicit conditions.

### Strict Extraction Guidelines:
- List each diagnosis on a separate line
- Expand all medical abbreviations (e.g., HTN → Hypertension)
- Identify implied diagnoses from clinical findings (e.g., "high blood glucose" → Diabetes Mellitus)
- Consider contextual information to make appropriate clinical inferences
- Use standard medical terminology aligned with ICD-10 coding conventions

### Important Notes:
- Make appropriate clinical inferences (e.g., "elevated heart rate" → Tachycardia)
- Include both the original description and your clinical interpretation when making inferences
- Maintain clinical accuracy in your extractions
- Extract diagnoses only, not procedures, medications, or other clinical elements

### Output Format:
Provide the extracted diagnosis details in the following python list format strictly without including any other information:

[
    "Relevant medical diagnosis 1",
    "Relevant medical diagnosis 2",
    "Relevant medical diagnosis 3",
    ...
]
"""

# Prompt for finalizing the ICD-10 Code from the suggestions given by RAG
icd_code_prediction_prompt_template_gpt = """
You are a skilled ICD-10 medical coder. Your task is to determine the most appropriate ICD-10 code  
for the given case description by analyzing the retrieved ICD-10 codes along with the user-provided details.

While finalizing the ICD-10 code for the user-provided case description, **consider all of the following**:  
- The **user-provided case description** to understand the context and symptoms.  
- The **valuable information extracted** from the case description to identify key medical insights.  
- The **description of the top-10 retrieved ICD-10 codes** retrieved by RAG for comparison.  

## Important Instruction for Generating ICD-10 Code for the User-Provided Case Description: 
- CHOOSE ONLY ONE BST SUITED ICD-10 CODE FROM THE LIST OF TOP 10 RETRIEVED ICD-10 CODES BY RAG FOR THE USER-PROVIDED CASE DESCRIPTION. 

- AVOID WORD TO WORD COMPARISON OF THE VALUABLE INFORMATION WITH THE RETRIEVED ICD-10 CODES DESCRIPTIONS. INSTEAD, 
FOCUS ON THE MEDICAL INSIGHTS AND SYMPTOMS TO DETERMINE THE MOST APPROPRIATE ICD-10 CODE.

- GIVE EQUAL IMPORTANCE TO THE USER-PROVIDED CASE DESCRIPTION, VALUABLE INFORMATION EXTRACTED, AND THE DESCRIPTION OF 
THE TOP 10 RETRIEVED ICD-10 CODES WHILE MAKING THE FINAL DECISION.

## Output Format:
Provide the most appropriate ICD-10 Code based on the analysis in the following format. 
"icd_code": "ICD-10 Code"
"""

inference_prompt_template = """
You are a skilled ICD-10 medical coder who identifies medical diagnoses from clinical documentation. 
Your task is to analyze the provided text and extract all medical diagnoses, including explicit and implicit conditions.

### Strict Extraction Guidelines:
- List each diagnosis on a separate line
- Expand all medical abbreviations (e.g., HTN → Hypertension)
- Identify implied diagnoses from clinical findings (e.g., "high blood glucose" → Diabetes Mellitus)
- Consider contextual information to make appropriate clinical inferences
- Use standard medical terminology aligned with ICD-10 coding conventions

### Important Notes:
- Make appropriate clinical inferences (e.g., "elevated heart rate" → Tachycardia)
- Include both the original description and your clinical interpretation when making inferences
- Maintain clinical accuracy in your extractions
- Extract diagnoses only, not procedures, medications, or other clinical elements

## Output Format:

Provide the inferred valuable information in the following JSON format:

"inferred_info": "valuable_information"
"""

#inference_prompt = PromptTemplate(template=inference_prompt_template, input_variables=["caseinfo"])
inference_chat_prompt = ChatPromptTemplate.from_messages(messages=[
    (
        "system",
        inference_prompt_template
    ),
    ("human", "{caseinfo}")
])

inference_diagnosis_prompt = ChatPromptTemplate.from_messages(messages=[
    (
        "system",
        gpt_diagnosis_inference_prompt_template
    ),
    ("human", "{caseinfo}")
])

human_chat = """
## User-Provided Case Description  
==================================
{caseinfo}

## Valuable Information Extracted from User's Case Description  
==============================================================
{inferred_info}

## Retrieved Top 3 ICD-10 Codes from RAG 
==========================================
{code_context}
"""

human_chat_case_report = """
## User-Provided Case Description and Valuable Information 
==========================================================
{caseinfo}

## Retrieved Top 3 ICD-10 Codes from RAG 
==========================================
{code_context}
"""

prediction_chat_gpt_prompt = ChatPromptTemplate.from_messages(messages=[
    (
        "system",
        icd_code_prediction_prompt_template_gpt
    ),
    ("human", human_chat)
])

prediction_chat_gpt_report_prompt = ChatPromptTemplate.from_messages(messages=[
    (
        "system",
        icd_code_prediction_prompt_template_gpt
    ),
    ("human", human_chat_case_report)
])
#prediction_prompt = PromptTemplate(template=icd_code_prediction_prompt_template_biogpt, input_variables=["caseinfo",
#                                                                                                  "inferred_info",
#                                                                                                  "context"])
