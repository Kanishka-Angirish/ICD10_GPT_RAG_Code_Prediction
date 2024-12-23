from bioGPT import causal_llm, causal_model
from utils import logger
from faiss_vector_store import FaissVectorStore
from embedding_generator import EmbeddingGenerator
import click
import pandas as pd


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
        for row in input_data.to_dict(orient="records"):
            faiss_vector_store.add_vector(row)
        logger.info("Knowledgebase Update Completed")
        # Save the Faiss Vector Store and Data
        faiss_vector_store.save_index()

    elif mode == "2":
        logger.info("Starting ICD Code Prediction")
        # Initialize the Embedding Generator
        embedding_generator = EmbeddingGenerator(causal_llm)
        # Initialize the Faiss Vector Store
        faiss_vector_store = FaissVectorStore(embedding_generator)
        logger.info("ICD Code Prediction Completed")
    else:
        logger.error("Invalid Mode Selected. Exiting Application")
        return


if __name__ == "__main__":
    biogpt_application()
