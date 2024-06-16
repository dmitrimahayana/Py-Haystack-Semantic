from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import DensePassageRetriever
from haystack.nodes import SentenceTransformersRanker
from Metadata_Filter import MetadataFilter
from dotenv import load_dotenv
from haystack import Document
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
MULTI_QA_MPNET_MODEL = os.getenv('multi_qa_mpnet_model')
DPR_QUESTION_MODEL = os.getenv('dpr_question_model')
DPR_CTX_MODEL = os.getenv('dpr_ctx_model')
MS_MARCO_MODEL = os.getenv('ms_marco_model')


def load_faiss_doc_store(db_path, faiss_index_path, faiss_config_path):
    if os.path.exists(faiss_index_path) and os.path.exists(faiss_config_path):
        document_store = FAISSDocumentStore(faiss_index_path=faiss_index_path,
                                            faiss_config_path=faiss_config_path)
    else:
        document_store = FAISSDocumentStore(sql_url=f"sqlite:///{db_path}",
                                            embedding_dim=768,
                                            faiss_index_factory_str="Flat")
    return document_store


def create_index(documents):
    print("Indexing Start...")
    # Define FAISS document Store
    document_store = load_faiss_doc_store(FAISS_DB_PATH, FAISS_INDEX_PATH, FAISS_CONFIG_PATH)

    # Define ES document store
    # document_store = ElasticsearchDocumentStore(host=ES_HOST,
    #                                             port=ES_PORT,
    #                                             scheme=ES_SCHEME,
    #                                             verify_certs=ast.literal_eval(ES_VERIFY_CERTS),
    #                                             ca_certs=ES_CA_CERTS,
    #                                             username=ES_USERNAME,
    #                                             password=ES_PASSWORD,
    #                                             embedding_dim=int(ES_EMBEDDING_DIM),
    #                                             index=ES_PREFIX_INDEX + type + ES_EMBEDDING_DIM)

    # Define Retriever
    # retriever1 = EmbeddingRetriever(document_store=document_store,
    #                                 embedding_model=MULTI_QA_MPNET_MODEL)  # multi-qa-mpnet-base-dot-v1 or all-mpnet-base-v2
    # embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")  # multi-qa-mpnet-base-dot-v1 or all-mpnet-base-v2
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=DPR_QUESTION_MODEL,
        passage_embedding_model=DPR_CTX_MODEL
        # query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        # passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )

    # Update Doc Embedding
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)
    document_store.save(FAISS_INDEX_PATH)  # Fais Doc Store
    print("Indexing Done...")


def perform_query(query_string, N, filters):
    print("Retrieval Start...")
    # Define FAISS document Store
    document_store = load_faiss_doc_store(FAISS_DB_PATH, FAISS_INDEX_PATH, FAISS_CONFIG_PATH)

    # Define ES document store
    # document_store = ElasticsearchDocumentStore(host=ES_HOST,
    #                                             port=ES_PORT,
    #                                             scheme=ES_SCHEME,
    #                                             verify_certs=ast.literal_eval(ES_VERIFY_CERTS),
    #                                             ca_certs=ES_CA_CERTS,
    #                                             username=ES_USERNAME,
    #                                             password=ES_PASSWORD,
    #                                             embedding_dim=int(ES_EMBEDDING_DIM),
    #                                             index=ES_PREFIX_INDEX + type + ES_EMBEDDING_DIM)

    # Define Retriever
    # retriever1 = EmbeddingRetriever(document_store=document_store,
    #                                 embedding_model=MULTI_QA_MPNET_MODEL)  # multi-qa-mpnet-base-dot-v1 or all-mpnet-base-v2
    # embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")  # multi-qa-mpnet-base-dot-v1 or all-mpnet-base-v2
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=DPR_QUESTION_MODEL,
        passage_embedding_model=DPR_CTX_MODEL
        # query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        # passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )

    # Define Ranker
    ranker = SentenceTransformersRanker(model_name_or_path=MS_MARCO_MODEL)
    # ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Create Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
    query_pipeline.add_node(component=MetadataFilter(), name="FilterMetadata", inputs=["Ranker"])

    # Perform Query
    top_k_result = N
    results = query_pipeline.run(query=query_string,
                                 params={
                                     "Retriever": {"top_k": 200},
                                     "Ranker": {"top_k": 200},
                                     "FilterMetadata": {"filter": filters, "top_k": top_k_result},
                                 })
    print("Query:", query_string)
    for row in results['documents']:
        print(f"ID: {row.id}, Content: {row.content[:100]}, Score: {row.score}")
    print("Retrieval Done...")


if __name__ == "__main__":
    # Load Data
    df = pd.read_csv("./Data/DigiDB_digimonlist.csv")
    # df['content'] = df.apply(lambda row: ', '.join([f"{index} {value}" for index, value in row.items()]), axis=1)
    df.loc[:, 'content'] = df['Digimon']
    documents_raw = df[['Number', 'content', 'Stage', 'Type', 'Attribute']].to_dict(orient='records')
    documents_final = [
        Document(
            content=data['content'],
            id=data['Number'],
            meta={
                "stage": data['Stage'],
                "type": data['Type'],
                "attribute": data['Attribute'],
            }
        ) for data in documents_raw]

    # Perform Indexing
    create_index(documents_final)

    # Perform Searching
    filters = {
        "attribute": "fire"
    }
    perform_query('greymon', 20, filters)
