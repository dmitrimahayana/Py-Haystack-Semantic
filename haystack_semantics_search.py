from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import DensePassageRetriever
from haystack.nodes import SentenceTransformersRanker
from haystack.nodes import FARMReader
from dotenv import load_dotenv
import pandas as pd
import os
import ast

# Define env variable
load_dotenv(".env")

# Define constants variable
FAISS_DB_PATH = os.getenv('faiss_db_path')
FAISS_INDEX_PATH = os.getenv('faiss_index_path')
FAISS_CONFIG_PATH = os.getenv('faiss_config_path')
ES_HOST = os.getenv('es_host')
ES_PORT = os.getenv('es_port')
ES_SCHEME = os.getenv('es_scheme')
ES_VERIFY_CERTS = os.getenv('es_verify_certs')
ES_CA_CERTS = os.getenv('es_ca_certs')
ES_USERNAME = os.getenv('es_username')
ES_PASSWORD = os.getenv('es_password')
ES_EMBEDDING_DIM = os.getenv('es_embedding_dim')
ES_PREFIX_INDEX = os.getenv('es_prefix_index')


def create_index(documents, type):
    # Delete Existing Faiss Store
    # if os.path.exists(FAISS_DB_PATH):
    #     os.remove(FAISS_DB_PATH)
    # if os.path.exists(FAISS_INDEX_PATH):
    #     os.remove(FAISS_INDEX_PATH)
    # if os.path.exists(FAISS_CONFIG_PATH):
    #     os.remove(FAISS_CONFIG_PATH)
    # Create Store
    # document_store = FAISSDocumentStore(sql_url=f"sqlite:///{FAISS_DB_PATH}",
    #                                     embedding_dim=768,  # 768 OR 1536
    #                                     faiss_index_factory_str="Flat")
    # Define ES document store
    document_store = ElasticsearchDocumentStore(host=ES_HOST,
                                                port=ES_PORT,
                                                scheme=ES_SCHEME,
                                                verify_certs=ast.literal_eval(ES_VERIFY_CERTS),
                                                ca_certs=ES_CA_CERTS,
                                                username=ES_USERNAME,
                                                password=ES_PASSWORD,
                                                embedding_dim=int(ES_EMBEDDING_DIM),
                                                index=ES_PREFIX_INDEX + type + ES_EMBEDDING_DIM)

    # Define Retriever
    # retriever = EmbeddingRetriever(document_store=document_store,
    #                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )

    # Update Doc Embedding
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)
    # document_store.save(FAISS_INDEX_PATH)


def perform_query(query_string, N, type):
    # Reload Document Store
    # document_store = FAISSDocumentStore(faiss_index_path=FAISS_INDEX_PATH,
    #                                     faiss_config_path=FAISS_CONFIG_PATH)\
    # Define ES document store
    document_store = ElasticsearchDocumentStore(host=ES_HOST,
                                                port=ES_PORT,
                                                scheme=ES_SCHEME,
                                                verify_certs=ast.literal_eval(ES_VERIFY_CERTS),
                                                ca_certs=ES_CA_CERTS,
                                                username=ES_USERNAME,
                                                password=ES_PASSWORD,
                                                embedding_dim=int(ES_EMBEDDING_DIM),
                                                index=ES_PREFIX_INDEX + type + ES_EMBEDDING_DIM)

    # Define Retriever
    # retriever = EmbeddingRetriever(document_store=document_store,
    #                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )

    # Define Ranker
    ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Define Reader
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # Create Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
    query_pipeline.add_node(component=reader, name="Reader", inputs=["Ranker"])

    # Perform Query
    top_k_retriever = N
    top_k_ranker = N
    top_k_reader = N

    results = query_pipeline.run(query=query_string,
                                 params=
                                 {"Retriever": {"top_k": top_k_retriever},
                                  "Ranker": {"top_k": top_k_ranker},
                                  "Reader": {"top_k": top_k_reader}
                                  })
    print("Query:", query_string)
    for row in results['documents']:
        print(f"ID: {row.id}, Content: {row.content[:100]}, Score: {row.score}")
    print("Done...")


def update_index(documents, type):
    # Reload Document Store
    # document_store = FAISSDocumentStore(faiss_index_path=FAISS_INDEX_PATH,
    #                                     faiss_config_path=FAISS_CONFIG_PATH)
    # Define ES document store
    document_store = ElasticsearchDocumentStore(host=ES_HOST,
                                                port=ES_PORT,
                                                scheme=ES_SCHEME,
                                                verify_certs=ast.literal_eval(ES_VERIFY_CERTS),
                                                ca_certs=ES_CA_CERTS,
                                                username=ES_USERNAME,
                                                password=ES_PASSWORD,
                                                embedding_dim=int(ES_EMBEDDING_DIM),
                                                index=ES_PREFIX_INDEX + type + ES_EMBEDDING_DIM)

    # Define Retriever
    # retriever = EmbeddingRetriever(document_store=document_store,
    #                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )

    # Update Doc Embedding
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)
    # document_store.save(FAISS_INDEX_PATH)


if __name__ == "__main__":
    # Load Data
    df = pd.read_csv("./Data/DigiDB_digimonlist.csv")
    df['content'] = df.apply(lambda row: ', '.join([f"{index} {value}" for index, value in row.items()]), axis=1)
    documents = df[['Number', 'content']].to_dict(orient='records')

    # Perform Indexing
    create_index(documents, 'digimon')

    # Perform Searching
    perform_query('Show me the list Digimon type Mega-Virus', 3, 'digimon')
