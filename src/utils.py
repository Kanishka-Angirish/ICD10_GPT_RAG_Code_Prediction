import logging
import os
import datetime
import sys

logdir = "../logging"
logging.basicConfig(format='%(asctime)s [%(name)s: %(lineno)s] - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO,
                    stream=sys.stdout)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("This is an info message")
logger.error("This is an error message")

vector_paths = {
    "faiss_index": "../vector_store/faiss_index.index",
    "data_store": "../vector_store/data_store.pkl"
}
