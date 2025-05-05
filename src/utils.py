import ast
import logging
import os
import datetime
import re
import pandas as pd
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
    "faiss_index": "../vector_store/faiss_index_biomedical.index",
    "data_store": "../resources/final_biomedical_data.csv",
    "test_data": "../resources/test_data.csv"
    # "data_store": "../vector_store/data_store.csv"
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")


def parse_context(context: dict,
                  data_type: str) -> str:
    if data_type == "biomedical_case":
        final_context = f"Case Description: {context['case_description']}\n" \
                        f"Valuable Information: {context['important_feature']} \n ======================"

    elif data_type == "biomedical_code":
        final_context = f"ICD-10 Code: {context['Code']}\n" \
                        f"Code Description: {context['Description']} \n ======================"

    elif data_type == "biomedical_report":
        final_context = f"Report: {context['original_text']} \n" \
                        f"Diagnosis: {context['billable_info']} \n================="

    else:
        final_context = "No context available"

    return final_context


def get_code_description(code: str):
    code_df = pd.read_csv("../resources/icd_dictionary.csv")
    matching_rows = code_df[code_df["Code"] == code]
    if matching_rows.empty:
        code_description = "No Description Found"
    else:
        code_description = matching_rows["Description"].values[0]
    return code_description


def extract_code_from_biogpt_response(prediction_response: dict, model_type: str) -> str:
    if model_type == "1":
        code = re.search(r'Based on this analysis, the most appropriate ICD-10 Code is:\s*(.*)',
                         prediction_response['text'])
    else:
        code = re.search(r'"icd_code":\s*"(.*)"', prediction_response['text'])

    if code:
        icd_code = re.split(r'\W+', code.group(1))[0].strip()
        return icd_code
    else:
        logger.error("ICD Code not found in the prediction response")
        return "N/A"


def prediction_accuracy(result_df: pd.DataFrame) -> float:
    result_df = result_df[~result_df["predicted_icd_code"].isin(["N/A", "OPENAI ERROR"])]
    correct_predictions = result_df[result_df["icd-10_code"] == result_df["predicted_icd_code"]]
    accuracy = len(correct_predictions) / len(result_df)
    return accuracy


def retrieve_rag_codes(code_context: str) -> list:
    code_list = re.findall(r"ICD-10 Code: (\w+)", code_context)
    return code_list


def generate_new_report_data(ground_truth, response) -> pd.DataFrame:
    #pattern = re.compile(r'\[(.*?)\]', re.DOTALL)
    #matches = [match.strip().replace('\n', '') for match in pattern.findall(response['text'])]
    #print(len(matches))
    # response = [match.strip() for match in matches]
    diagnosis = ast.literal_eval(response)
    print(f"Diagnosis for report \n {diagnosis}")

    data = pd.DataFrame({"case_description": diagnosis, "ground_truth": ground_truth})
    #data.to_csv(f"../resources/test_data_report_diagnosis_{index}.csv", index=False)
    return data
