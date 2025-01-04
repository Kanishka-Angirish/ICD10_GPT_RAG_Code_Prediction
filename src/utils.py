import logging
import os
import datetime
import sys
from dotenv import load_dotenv

logdir = "../logging"
logging.basicConfig(format='%(asctime)s [%(name)s: %(lineno)s] - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO,
                    stream=sys.stdout)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("This is an info message")
logger.error("This is an error message")

load_dotenv()

vector_paths = {
    "faiss_index": "../vector_store/faiss_index.index",
    "data_store": "../resources/final_biomedical_data.csv"
    # "data_store": "../vector_store/data_store.csv"
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")


def parse_context(context: dict) -> str:
    final_context = f"Case Description: {context['case_description']}\n" \
                    f"Valuable Information: {context['important_feature']} \n" \
                    f"ICD-10 Code: {context['icd-10_code']} \n ======================"

    return final_context
