import os.path

import faiss
import numpy as np
import pandas as pd

from embedding_generator import EmbeddingGenerator
from utils import logger, vector_paths, parse_context
from tqdm import tqdm


class FaissVectorStore:

    def __init__(self, embedding_generator: EmbeddingGenerator,
                 data_type: str = "biomedical_case",
                 vector_path: str = None,
                 data_store_path: str = None) -> None:

        if vector_path is None:
            self.vector_path = vector_paths['faiss_index']
        else:
            self.vector_path = vector_path

        self.data_type = data_type

        if data_store_path is None:
            self.data_store_path = vector_paths['data_store']
        else:
            self.data_store_path = data_store_path

        self.embedding_generator = embedding_generator

        if os.path.exists(self.vector_path):
            # Referring to the updated Vector Store
            self.index = faiss.read_index(self.vector_path)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_generator.get_embedding_dimensions())

        if os.path.exists(self.data_store_path):
            self.data_store = pd.read_csv(self.data_store_path)
        else:
            logger.info("Data store not found. Creating a new data store")
            self.data_store = pd.DataFrame()

    def add_vector(self, data: pd.DataFrame) -> None:

        logger.info("Adding vector to the vector store")
        embeddings = []

        for _, row in tqdm(data.iterrows()):
            if self.data_store_path == vector_paths['data_store']:
                case_info = "Case Description: {}".format(row["case_description"])
            elif self.data_type == "biomedical_code":
                case_info = row["Description"]
            else:
                case_info = row["original_text"]

            embedding = self.embedding_generator.generate_embeddings(case_info)

            # Ensure embedding is a NumPy array
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            if embedding.shape[-1] != self.index.d:
                logger.error(f"Embedding dimension mismatch: {embedding.shape[-1]} != {self.index.d}")
                return
            embeddings.append(embedding)

        embeddings = np.vstack(embeddings)
        self.index.add(embeddings)
        logger.info("Adding embedding corresponding records to the data store")
        if not self.data_store.equals(data):
            self.data_store = pd.concat([self.data_store, data], ignore_index=True)

    def retrieve_context(self, query, top_k=3):
        logger.info(f"Retrieving context for query: {query}")

        query_embedding = self.embedding_generator.generate_embeddings(query)

        # Ensure embedding is a NumPy array
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        query_embedding_np = query_embedding.astype('float32').reshape(1, -1)
        distances, index = self.index.search(query_embedding_np, top_k)
        logger.info(f"Top {top_k} results: {index}")
        context = "\n".join([parse_context(row, self.data_type) for row in self.data_store.iloc[index[0], :].to_dict(orient="records")])
        return context

    def save_index(self, index_path="", data_path=""):

        if index_path == "":
            index_path = self.vector_path
        if data_path == "":
            data_path = self.data_store_path

        logger.info(f"Saving index to {index_path}")
        faiss.write_index(self.index, index_path)

        logger.info(f"Saving data store to {data_path}")
        self.data_store.to_csv(data_path, index=False)
