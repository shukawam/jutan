import logging, os
from dotenv import find_dotenv, load_dotenv
from typing import List
import faiss
import oracledb
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import OracleVS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere.rerank import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

_ = load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


def _get_embedding_function(
    model_name: str = "openai",
) -> OpenAIEmbeddings | CohereEmbeddings:
    """Initialize vector store."""
    if "openai".__eq__(model_name):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_function = OpenAIEmbeddings(
            api_key=openai_api_key, model="text-embedding-ada-002"
        )
        logger.info("Use OpenAI Embedding model")
    elif "cohere".__eq__(model_name):
        cohere_api_key = os.getenv("COHERE_API_KEY")
        embedding_function = CohereEmbeddings(
            cohere_api_key=cohere_api_key, model="embed-multilingual-v3.0"
        )
        logger.info("Use Cohere Embedding model")
    elif "oci".__eq__(model_name):
        compartment_id = os.getenv("COMPARTMENT_ID")
        embedding_function = OCIGenAIEmbeddings(
            auth_type="INSTANCE_PRINCIPAL",
            compartment_id=compartment_id,
            service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
            model_id="cohere.embed-multilingual-v3.0",
        )
        logger.info("Use OCI Embedding model")
    else:
        logger.error("Unsetted model name")
    return embedding_function


def _load_documents() -> List[Document]:
    dataset = os.getenv("DATASET")
    csv_loader = CSVLoader(file_path=f"../data/{dataset}/data.csv")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False,
    )
    documents = csv_loader.load_and_split(text_splitter=splitter)
    return documents


class VectorStore:
    def __init__(
        self, model_name: str = "openai", vector_store_option: str = "faiss", **kwargs
    ) -> None:
        embedding_functions = _get_embedding_function(model_name=model_name)
        if "faiss".__eq__(vector_store_option):
            index = faiss.IndexFlatL2(
                len(embedding_functions.embed_query("for checking dimensions."))
            )
            if os.path.isdir("faiss_index"):
                # ローカルのセーブポイントがあるなら使用する
                vector_store = FAISS.load_local(
                    folder_path="faiss_index",
                    embeddings=embedding_functions,
                    allow_dangerous_deserialization=True,
                )
            else:
                # ローカルのセーブポイントがない場合
                vector_store = FAISS(
                    embedding_function=embedding_functions,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
                documents = _load_documents()
                vector_store.add_documents(documents=documents)
                vector_store.save_local(folder_path="faiss_index")
                self.vector_store = vector_store
                logger.info("Set vector store option to faiss")
        elif "oraclevs".__eq__(vector_store_option):
            connection = oracledb.connect(
                user=kwargs.get("user"),
                password=kwargs.get("password"),
                dsn=kwargs.get("dsn"),
                config_dir=kwargs.get("config_dir"),
                wallet_location=kwargs.get("wallet_location"),
                wallet_password=kwargs.get("wallet_password"),
            )
            vector_store = OracleVS(
                client=connection,
                embedding_function=embedding_functions,
                table_name="NVIDIA_AI_SUMMIT_SESSIONS",
                distance_strategy=DistanceStrategy.COSINE,
            )
            documents = _load_documents()
            vector_store.add_documents(documents=documents)
            self.vector_store = vector_store
            logger.info("Set vector store option to oraclevs")
        else:
            logger.error("Unseted vector store option.")

    def get_retriever(
        self, top_k: int = 25, use_reranker: bool = True, top_n: int = 5
    ) -> VectorStoreRetriever:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        if use_reranker == True:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            compressor = CohereRerank(
                cohere_api_key=cohere_api_key,
                model="rerank-multilingual-v3.0",
                top_n=top_n,
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            return compression_retriever
        else:
            return retriever
