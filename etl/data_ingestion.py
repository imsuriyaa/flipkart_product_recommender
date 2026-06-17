import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from logger import GLOBAL_LOGGER as log

class DataIngestion:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """

    def __init__(self):
        """
        Initialize environment variables, embedding model, and set CSV file path.
        """
        print("Initializing DataIngestion pipeline...")
        self.model_loader=ModelLoader()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config=load_config()
       

    def _get_csv_path(self):
        """
        Get path to the CSV file located inside 'data' folder.
        """
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir,'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path

    def _load_csv(self):
        """
        Load product data from CSV.
        """
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id','product_title', 'rating', 'total_reviews','price', 'top_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    def transform_data(self):
        """
        Transform product data into list of LangChain Document objects.
        """
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                    "product_id": row["product_id"],
                    "product_title": row["product_title"],
                    "rating": row["rating"],
                    "total_reviews": row["total_reviews"],
                    "price": row["price"],
                    "top_reviews": row["top_reviews"]
                }
            product_list.append(product_entry)

        documents = []
        for entry in product_list:
            metadata = {
                    "product_id": entry["product_id"],
                    "product_title": entry["product_title"],
                    "rating": entry["rating"],
                    "total_reviews": entry["total_reviews"],
                    "price": entry["price"]
            }
            doc = Document(page_content=entry["top_reviews"], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """
        Store documents into ChromaDB vector store.
        """

        vstore = Chroma(
            collection_name='example_collection',
            embedding_function=self.model_loader.load_embeddings(),
            persist_directory="./chroma_db",
        )

        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids

    def run_pipeline(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        log.info("Starting data ingestion pipeline...")
        documents = self.transform_data()

        log.info("Storing documents in vector database...")
        vstore, _ = self.store_in_vector_db(documents)

        #Optionally do a quick search
        query = "Can you tell me the low budget iphone?"
        results = vstore.similarity_search(query)

        log.info(f"Sample search results for query: '{query}'")
        for res in results:
            log.info(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

# Run if this file is executed directly
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()