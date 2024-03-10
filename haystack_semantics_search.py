from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import DensePassageRetriever
from haystack.nodes import SentenceTransformersRanker
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

# Define ES document store
document_store = ElasticsearchDocumentStore(host=ES_HOST,
                                            port=ES_PORT,
                                            scheme=ES_SCHEME,
                                            verify_certs=ast.literal_eval(ES_VERIFY_CERTS),
                                            ca_certs=ES_CA_CERTS,
                                            username=ES_USERNAME,
                                            password=ES_PASSWORD,
                                            embedding_dim=int(ES_EMBEDDING_DIM),
                                            index=ES_PREFIX_INDEX + '_fb_' + ES_EMBEDDING_DIM)


def create_index(documents):
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


def perform_query(query_string, N):
    # Reload Document Store
    # document_store = FAISSDocumentStore(faiss_index_path=FAISS_INDEX_PATH,
    #                                     faiss_config_path=FAISS_CONFIG_PATH)\

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

    # Create Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])

    # Perform Query
    top_k_retriever = N
    top_k_ranker = N

    results = query_pipeline.run(query=query_string,
                                 params={"Retriever": {"top_k": top_k_retriever}, "Ranker": {"top_k": top_k_ranker}})
    print("Query:", query_string)
    for row in results['documents']:
        print(f"ID: {row.id}, Content: {row.content[:100]}, Score: {row.score}")
    print("Done...")


def update_index(dict_faiss_config, documents):
    # Reload Document Store
    # document_store = FAISSDocumentStore(faiss_index_path=FAISS_INDEX_PATH,
    #                                     faiss_config_path=FAISS_CONFIG_PATH)\

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
    # Sample Data
    data = {
        'id': [10001, 10002, 10003, 10004, 10005],
        'url': ['https://docs.haystack.deepset.ai/docs/10001', 'https://docs.haystack.deepset.ai/docs/10002',
                'https://docs.haystack.deepset.ai/docs/10003', 'https://docs.haystack.deepset.ai/docs/10004',
                'https://docs.haystack.deepset.ai/docs/10005'],
        'header': ['semantic search', 'search', 'semantic engine', 'ultimate semantic', 'search engine'],
        'body': ['Haystack is an open-source framework for building production-ready LLM applications',
                 'retrieval-augmented generative pipelines and state-of-the-art search systems that work intelligently over large document collections',
                 'Haystack is an end-to-end framework that you can use to build powerful and production-ready pipelines with Large Language Models (LLMs) for different search use cases',
                 'Whether you want to perform retrieval-augmented generation (RAG), question answering, or semantic document search, you can use the state-of-the-art LLMs and NLP models in Haystack to provide custom search experiences and make it possible for your users to query in natural language',
                 'Haystack is built in a modular fashion so that you can combine the best technology from OpenAI, Cohere, SageMaker, and other open source projects, like Hugging Face\'s Transformers, Elasticsearch, or Milvus']
    }
    df = pd.DataFrame(data)
    df['content'] = df['header'] + ' ' + df['body']
    documents = df[['id', 'content']].to_dict(orient='records')

    # Perform Indexing
    create_index(documents)

    # Perform Searching
    perform_query('Haystack', 3, )
