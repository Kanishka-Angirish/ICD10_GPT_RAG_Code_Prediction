import ast

from gptmodels import causal_model, llm_model, embedding_model, chat_model
from utils import logger, vector_paths, get_code_description, extract_code_from_biogpt_response, \
    prediction_accuracy, retrieve_rag_codes, generate_new_report_data
from faiss_vector_store import FaissVectorStore
from embedding_generator import EmbeddingGenerator
import click
from tqdm import tqdm
import datetime
import os
import pandas as pd
from prompts import inference_chat_prompt, prediction_chat_gpt_prompt, \
    prediction_chat_gpt_report_prompt, inference_diagnosis_prompt
from langchain.chains.llm import LLMChain


def icd_code_prediction(case_info: str,
                        model_type: str) -> tuple:
    logger.info("Starting ICD Code Prediction")
    # Initialize the Embedding Generator
    embedding_generator = EmbeddingGenerator(embedding_model)
    # Initialize the Faiss Vector Store
    faiss_vector_store = FaissVectorStore(embedding_generator)
    retrieval_query = f"Case Description: {case_info}"
    additional_context = faiss_vector_store.retrieve_context(query=retrieval_query)
    logger.info(f"Additional Context: \n {additional_context}")
    first_prompt_chain = LLMChain(llm=llm_model, prompt=inference_chat_prompt)
    # Try block to handle openai Bad Request Error
    try:
        response = first_prompt_chain.invoke(input={"caseinfo": case_info, "additional_context": additional_context})
        valuable_information = response["text"]
    except Exception as e:
        logger.error(f"Error while invoking Azure OpenAI Model: {e}")
        return "OPENAI ERROR", None

    # Initialize the Faiss Vector Store for ICD Code Dictionary
    logger.info("Retrieving ICD Code Context")
    code_faiss_vector_store = FaissVectorStore(embedding_generator,
                                               vector_path="../vector_store/faiss_index_code_azure.index",
                                               data_store_path="../resources/icd_dictionary.csv",
                                               data_type="biomedical_code")
    code_context = code_faiss_vector_store.retrieve_context(query=valuable_information, top_k=10)
    rag_codes = retrieve_rag_codes(code_context)
    logger.info(f"ICD Code Context: \n {code_context}")
    #if model_type == "1":
    #    second_prompt_chain = LLMChain(llm=causal_llm, prompt=prediction_prompt)
    if model_type == "1":
        second_prompt_chain = LLMChain(llm=chat_model, prompt=prediction_chat_gpt_prompt)
    elif model_type == "2":
        second_prompt_chain = LLMChain(llm=llm_model, prompt=prediction_chat_gpt_prompt)
    else:
        raise ValueError("Invalid model type selected. Please choose 1 or 2")

    prediction_response = second_prompt_chain.invoke(input={"caseinfo": case_info,
                                                            "inferred_info": valuable_information,
                                                            "code_context": code_context})
    # logger.info(f"ICD Code Prediction Response: {prediction_response}")
    # logger.info(f"ICD Code Prediction Response Keys: {prediction_response.keys()}")
    logger.info(f"ICD Code Prediction Response Text: {prediction_response['text']}")

    return extract_code_from_biogpt_response(prediction_response, model_type=model_type), rag_codes


def icd_code_prediction_report(case_info: str, model_type: str) -> tuple:
    logger.info("Starting ICD Code Prediction")
    # Initialize the Embedding Generator
    embedding_generator = EmbeddingGenerator(embedding_model)

    logger.info("Retrieving ICD Code Context")
    code_faiss_vector_store = FaissVectorStore(embedding_generator,
                                               vector_path="../vector_store/faiss_index_code_azure.index",
                                               data_store_path="../resources/icd_dictionary.csv",
                                               data_type="biomedical_code")

    code_context = code_faiss_vector_store.retrieve_context(query=case_info, top_k=10)
    rag_codes = retrieve_rag_codes(code_context)
    logger.info(f"ICD Code Context: \n {code_context}")
    #if model_type == "1":
        # Not needed any longer (Deprecated)
        #second_prompt_chain = LLMChain(llm=causal_llm, prompt=prediction_chat_gpt_report_prompt)
    if model_type == "1":
        second_prompt_chain = LLMChain(llm=chat_model, prompt=prediction_chat_gpt_report_prompt)
    elif model_type == "2":
        second_prompt_chain = LLMChain(llm=llm_model, prompt=prediction_chat_gpt_report_prompt)
    else:
        raise ValueError("Invalid model type selected. Please choose 1 or 2")

    prediction_response = second_prompt_chain.invoke(input={"caseinfo": case_info,
                                                            "code_context": code_context})
    # logger.info(f"ICD Code Prediction Response: {prediction_response}")
    # logger.info(f"ICD Code Prediction Response Keys: {prediction_response.keys()}")
    logger.info(f"ICD Code Prediction Response Text: {prediction_response['text']}")

    return extract_code_from_biogpt_response(prediction_response, model_type=model_type), rag_codes


def _generate_report_diagnosis(case_info: str) -> str:
    logger.info("Generating Report Diagnosis")
    """
    # Initialize the Embedding Generator
    embedding_generator = EmbeddingGenerator(embedding_model)
    # Initialize the Faiss Vector Store
    report_faiss_vector_store = FaissVectorStore(embedding_generator,
                                                 vector_path="../vector_store/faiss_index_report_azure.index",
                                                 data_store_path="../resources/top_80_reports.csv",
                                                 data_type="biomedical_report")

    retrieval_query = f"{case_info}"
    additional_context = report_faiss_vector_store.retrieve_context(query=retrieval_query, top_k=1)
    logger.info(f"Additional Context: \n {additional_context}")
    """
    diagnosis_prompt_chain = LLMChain(llm=chat_model, prompt=inference_diagnosis_prompt)
    try:
        response = diagnosis_prompt_chain.invoke(
            input={"caseinfo": case_info})
        diagnosis_information = response["text"]
        return diagnosis_information
    except Exception as e:
        logger.error(f"Error while invoking Azure OpenAI Model: {e}")
        return "OPENAI ERROR", None


def _knowledge_base_creation(default_path: str, embedding_generator: EmbeddingGenerator,
                             vector_path: str = None, data_store_path: str = None,
                             data_type: str = "biomedical_case") -> None:
    # Initialize the Faiss Vector Store
    input_data_path = click.prompt("Enter the path of data to be added to the knowledgebase",
                                   type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
                                   default=default_path)
    faiss_vector_store = FaissVectorStore(embedding_generator, data_store_path=data_store_path,
                                          vector_path=vector_path, data_type=data_type)
    input_data = pd.read_csv(input_data_path)
    # Add the data to the Faiss Vector Store
    logger.info("Adding Data to the Faiss Vector Store")
    faiss_vector_store.add_vector(data=input_data)
    logger.info("Knowledgebase Update Completed")
    # Save the Faiss Vector Store and Data
    faiss_vector_store.save_index()


@click.command()
@click.option("--mode", prompt="Choose Mode \n 1. Knowledgebase Update \n 2. ICD Code Prediction",
              type=click.Choice(["1", "2"]), help="Choose the mode of operation")
def biogpt_application(mode):
    """
    This function is the entry point for the BioGPT Application
    """
    logger.info("Starting BioGPT Application")
    logger.info(f"Selected Mode: {mode}")

    if mode == "1":
        logger.info("Starting Knowledgebase Update")

        knowledgebase_option = click.prompt(
            "Choose the Knowledgebase Update Option \n 1. Add Case Description \n 2. Add Code Dictionary \n 3. Add Patient Report \n 4. All of the above",
            type=click.Choice(["1", "2", "3", "4"])
        )
        # Initialize the Embedding Generator
        embedding_generator = EmbeddingGenerator(embedding_model)

        # For creating vector store of case description data (Experimentation Only - Not to be used in Production)
        if knowledgebase_option in ["1", "4"]:
            logger.info("Updating Knowledgebase with Case Description Data")
            _knowledge_base_creation(default_path="../resources/final_biomedical_data.csv",
                                     embedding_generator=embedding_generator)

        # For creating vector store of code dictionary data
        if knowledgebase_option in ["2", "4"]:
            logger.info("Updating Knowledgebase with Code Dictionary Data")
            _knowledge_base_creation(default_path="../resources/icd_dictionary.csv",
                                     embedding_generator=embedding_generator,
                                     vector_path="../vector_store/faiss_index_code_azure.index",
                                     data_store_path="../resources/icd_dictionary.csv",
                                     data_type="biomedical_code")

        # For creating vector store of patient report data
        if knowledgebase_option in ["3", "4"]:
            logger.info("Updating Knowledgebase with Patient Report Data")
            _knowledge_base_creation(default_path="../resources/diagnosis_data_training.csv",
                                     embedding_generator=embedding_generator,
                                     vector_path="../vector_store/faiss_index_diagnosis_azure.index",
                                     data_store_path="../resources/diagnosis_data_training.csv",
                                     data_type="biomedical_report")

    elif mode == "2":
        test_data_filepath = click.prompt("Enter the name of the file to read data",
                                          type=click.STRING,
                                          default=vector_paths['test_data'])
        test_data = pd.read_csv(test_data_filepath)
        result = []
        rag_result = []
        run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        result_dir = click.prompt("Enter the path of directory to save results",
                                  type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
                                  default="../resources/result_dir")
        result_filename = click.prompt("Enter the name of the file to save results",
                                       type=click.STRING,
                                       default="result_biogpt.csv")
        model_type = click.prompt("Choose the Model for ICD Code Prediction \n 1. GPT-4 32K  \n 2. "
                                  "Azure OpenAI (GPT-35 Turbo)",
                                  type=click.Choice(["1", "2", "3"]))
        prediction_type = click.prompt("Choose the Prediction Type \n 1. Case Prediction \n 2. Prediction with Report",
                                       type=click.Choice(["1", "2"]))
        result_filename = result_filename.split(".")[0] + ".csv"
        result_path = os.path.join(result_dir, result_filename)

        if prediction_type == "2":
            report = test_data.loc[0, "original_text"]
            ground_truth = test_data.loc[0, "label"]
            diagnosis = _generate_report_diagnosis(report)
            test_data = generate_new_report_data(ground_truth=ground_truth, response=diagnosis)

        for _, data in tqdm(test_data.iloc[:201].iterrows()):
            case_info = data["case_description"]
            logger.info(f"User Case Description: {case_info}")
            # logger.info("Groundtruth Valuable Information: {}".format(data["important_feature"]))
            if prediction_type == "1":
                logger.info("Starting ICD Code Prediction")
                icd_code, rag_codes = icd_code_prediction(case_info, model_type)
            elif prediction_type == "2":
                logger.info("Starting ICD Code Prediction with Report")
                icd_code, rag_codes = icd_code_prediction_report(case_info, model_type)
            else:
                logger.error("Invalid Prediction Type Selected. Exiting Application")
                return
            logger.info(f"Generated ICD-10 Code: {icd_code}")
            result.append(icd_code)
            rag_result.append(rag_codes)

        ground_truth = test_data.iloc[:201].copy()
        ground_truth['predicted_icd_code'] = result
        ground_truth['rag_codes'] = rag_result
        ground_truth['predicted_icd_code_description'] = ground_truth['predicted_icd_code'].apply(
            lambda x: get_code_description(x))
        ground_truth.to_csv(result_path, index=False)
        # logger.info(f"Estimated Model Accuracy: {prediction_accuracy(ground_truth) * 100} %")
    else:
        logger.error("Invalid Mode Selected. Exiting Application")
        return


if __name__ == "__main__":
    biogpt_application()
