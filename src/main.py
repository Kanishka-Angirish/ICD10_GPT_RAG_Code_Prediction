from bioGPT import causal_llm, causal_model, llm_model
from utils import logger
from faiss_vector_store import FaissVectorStore
from embedding_generator import EmbeddingGenerator
import click
import pandas as pd
from prompts import inference_prompt, inference_chat_prompt, prediction_chat_prompt, prediction_prompt
from langchain.chains.llm import LLMChain


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
        # Initialize the Embedding Generator
        embedding_generator = EmbeddingGenerator(causal_model)
        # Initialize the Faiss Vector Store
        input_data_path = click.prompt("Enter the path of data to be added to the knowledgebase",
                                  type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
                                  default="../resources/final_biomedical_data.csv")
        faiss_vector_store = FaissVectorStore(embedding_generator)
        input_data = pd.read_csv(input_data_path)
        # Add the data to the Faiss Vector Store
        logger.info("Adding Data to the Faiss Vector Store")
        faiss_vector_store.add_vector(data=input_data)
        logger.info("Knowledgebase Update Completed")
        # Save the Faiss Vector Store and Data
        faiss_vector_store.save_index()

    elif mode == "2":
        logger.info("Starting ICD Code Prediction")
        # Initialize the Embedding Generator
        embedding_generator = EmbeddingGenerator(causal_model)
        # Initialize the Faiss Vector Store
        faiss_vector_store = FaissVectorStore(embedding_generator)
        first_prompt_chain = LLMChain(llm=llm_model, prompt=inference_chat_prompt)
        case_info = """Patient is a 29-year-old male presenting for follow-up of a displaced fracture of the second 
        metatarsal bone in the left foot. This is the second encounter since the injury occurred six weeks ago. 
        Radiographic imaging shows continued displacement of the fracture fragments and delayed healing, with callus 
        formation still absent. Patient reports mild pain and swelling localized to the midfoot. Plan includes 
        continued immobilization and consideration of surgical intervention if no healing is evident on next 
        follow-up in four weeks."""
        response = first_prompt_chain.invoke(input={"caseinfo": case_info})
        valuable_information = response["text"]
        retrieval_query = f"Case Description: {case_info} \n Valuable Information: {valuable_information}"
        additional_context = faiss_vector_store.retrieve_context(query=retrieval_query)
        #logger.info(f"Inference Query Response: {response['text']}")
        logger.info(f"Inference Query Response Keys: {response.keys()}")
        logger.info(f"Additional Context: {additional_context}")
        second_prompt_chain = LLMChain(llm=causal_llm, prompt=prediction_prompt)
        prediction_response = second_prompt_chain.invoke(input={"caseinfo": case_info,
                                                                "inferred_info": valuable_information,
                                                                "context": additional_context})
        logger.info(f"ICD Code Prediction Response: {prediction_response}")
        logger.info(f"ICD Code Prediction Response Keys: {prediction_response.keys()}")
        logger.info(f"ICD Code Prediction: {prediction_response['text']}")
    else:
        logger.error("Invalid Mode Selected. Exiting Application")
        return


if __name__ == "__main__":
    biogpt_application()
