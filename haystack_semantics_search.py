from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import DensePassageRetriever
from haystack.nodes import SentenceTransformersRanker
import pandas as pd
import os

def create_index(documents, dict_faiss_config):
    # Create Index
    if os.path.exists(dict_faiss_config['db_path']):
        os.remove(dict_faiss_config['db_path'])
    if os.path.exists(dict_faiss_config['faiss_index_path']):
        os.remove(dict_faiss_config['faiss_index_path'])
    if os.path.exists(dict_faiss_config['faiss_config_path']):
        os.remove(dict_faiss_config['faiss_config_path'])
    # Define Document Store
    # document_store = FAISSDocumentStore(sql_url=f"sqlite:///{dict_faiss_config['db_path']}",
    #                                     # embedding_dim=1536,
    #                                     faiss_index_factory_str="Flat")
    document_store = ElasticsearchDocumentStore(host='localhost',
                                                port=9200,
                                                scheme='https',
                                                verify_certs=True,
                                                ca_certs='D:/00 Project/00 My Project/docker-elasticsearch/certs/ca/ca.crt',
                                                api_key_id='-I-Nqo0BPuQqx5wtsBo8',
                                                api_key='brDDVnBiR7yti6kdeYBZBw',
                                                embedding_dim = 768, #768 or 1536,
                                                # index='1536-chatgpt-haystack-search',
                                                index='768-fb-haystack-search',
                                                )

    # Define Retriever
    # retriever = EmbeddingRetriever(document_store=document_store,
    #                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )
    # retriever = EmbeddingRetriever(
    #     document_store=document_store,
    #     batch_size=8,
    #     embedding_model="text-embedding-ada-002",
    #     api_key=os.environ.get('CHATGPT_API_KEY'),
    #     max_seq_len=1536
    # )

    # Update Doc Embedding
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)
    # document_store.save(dict_faiss_config['faiss_index_path'])


def perform_query(query_string, N, dict_faiss_config):
    # Reload Document Indexing
    # document_store = FAISSDocumentStore(faiss_index_path=dict_faiss_config['faiss_index_path'],
    #                                     faiss_config_path=dict_faiss_config['faiss_config_path'])
    document_store = ElasticsearchDocumentStore(host='localhost',
                                                port=9200,
                                                scheme='https',
                                                verify_certs=True,
                                                ca_certs='D:/00 Project/00 My Project/docker-elasticsearch/certs/ca/ca.crt',
                                                api_key_id='-I-Nqo0BPuQqx5wtsBo8',
                                                api_key='brDDVnBiR7yti6kdeYBZBw',
                                                embedding_dim = 768, #768 or 1536
                                                # index='1536-chatgpt-haystack-search',
                                                index='768-fb-haystack-search',
                                                )

    # Retriever Config
    # retriever = EmbeddingRetriever(document_store=document_store,
    #                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )
    # retriever = EmbeddingRetriever(
    #     document_store=document_store,
    #     batch_size=8,
    #     embedding_model="text-embedding-ada-002",
    #     api_key=os.environ.get('CHATGPT_API_KEY'),
    #     max_seq_len=1536
    # )

    # Define Ranker
    ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
    # ranker = LostInTheMiddleRanker(
    #     word_count_threshold=1024,
    #     top_k=N,
    # )

    # Create Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])

    # Perform Query
    top_k_retriever = N
    top_k_ranker = N

    # results = query_pipeline.run(query=query_string,
    #                              params={"Retriever": {"top_k": top_k_retriever}, "Ranker": {"top_k": top_k_ranker}})
    # results = query_pipeline.run(query=query_string,
    #                              params={"Ranker": {"top_k": top_k_ranker}})
    results = query_pipeline.run(query=query_string)
    print("Query:", query_string)
    for row in results['documents']:
        print(f"ID: {row.id}, Content: {row.content}")
    print("Done...")


# def update_index(dict_faiss_config, documents):
#     # Reload Document Indexing
#     document_store = FAISSDocumentStore(faiss_index_path=dict_faiss_config['faiss_index_path'],
#                                         faiss_config_path=dict_faiss_config['faiss_config_path'])
#     # document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")

#     # Define Retriever
#     retriever = EmbeddingRetriever(document_store=document_store,
#                                    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

#     # Update Doc Embedding
#     document_store.write_documents(documents)
#     document_store.update_embeddings(retriever)
#     document_store.save(dict_faiss_config['faiss_index_path'])


if __name__ == "__main__":
    # Document Store Config
    dict_faiss_config = {
        'db_path': 'faiss/faiss_dataframe_document_store.db',
        'faiss_index_path': 'faiss/faiss_dataframe_document_store.faiss',
        'faiss_config_path': 'faiss/faiss_dataframe_document_store.json'
    }

    # Sample DataFrame 1
    data = {
        'id': [1, 2, 3, 4, 5],
        'url': ['url1', 'url2', 'url3', 'url4', 'url5'],
        'title': ['apar', 'cek fatigue', 'microsleep', 'check fatigue', 'lingkungan berdebu'],
        'description': ['Apar Low Pressure unit CO2470',
                        'pemeriksaan kebugaran dan keamanan unit',
                        'microsleep fatigue',
                        'management fatigue',
                        'jalan hauling berdebu'],
        'location': ['loc1', 'loc2', 'loc3', 'loc4', 'loc5']
    }
    df = pd.DataFrame(data)
    df['content'] = df['title'] + ' ' + df['description']
    documents1 = df[['id', 'content']].to_dict(orient='records')

    # Call Indexing
    create_index(documents1, dict_faiss_config)

    # Call Search
    # perform_query('fatigue adult', 3, dict_faiss_config)
    perform_query('fatigguee', 3, dict_faiss_config)

    # # Sample DataFrame 2
    # data = {
    #     'id': [6, 7, 8, 9, 10],
    #     'url': ['url6', 'url7', 'url8', 'url9', 'url10'],
    #     'title': ['apar', 'cek fatigue', 'microsleep', 'check fatigue', 'lingkungan berdebu'],
    #     'description': ['Apar Low Pressure unit CO2470',
    #                     'pemeriksaan kebugaran dan keamanan unit',
    #                     'microsleep fatigue',
    #                     'management fatigue',
    #                     'jalan hauling berdebu'],
    #     'location': ['loc6', 'loc7', 'loc8', 'loc9', 'loc10']
    # }
    # df = pd.DataFrame(data)
    # df['content'] = df['title'] + ' ' + df['description']
    # documents2 = df[['id', 'content']].to_dict(orient='records')
    #
    # # Update Indexing
    # update_index(dict_faiss_config, documents2)
    #
    # # Call Search
    # perform_query('fatigue', 6, dict_faiss_config)
    # perform_query('fattigue', 6, dict_faiss_config)
