import pandas as pd
from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import os


def create_index(documents, dict_faiss_config):
    if os.path.exists(dict_faiss_config['db_path']):
        os.remove(dict_faiss_config['db_path'])
    if os.path.exists(dict_faiss_config['faiss_index_path']):
        os.remove(dict_faiss_config['faiss_index_path'])
    if os.path.exists(dict_faiss_config['faiss_config_path']):
        os.remove(dict_faiss_config['faiss_config_path'])
    document_store = FAISSDocumentStore(sql_url=f"sqlite:///{dict_faiss_config['db_path']}")

    # Retriever Config
    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

    # Update Doc Embedding
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)
    document_store.save(dict_faiss_config['faiss_index_path'])


def perform_query(query_string, dict_faiss_config):
    # Reload Document Indexing
    document_store = FAISSDocumentStore(
        faiss_index_path=dict_faiss_config['faiss_index_path'],
        faiss_config_path=dict_faiss_config['faiss_config_path'])

    # Retriever Config
    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

    # Create Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    # Perform Query
    results = query_pipeline.run(query=query_string, params={"Retriever": {"top_k": 3}})
    print("Query:", query_string)
    for row in results['documents']:
        print(f"ID: {row.id}, Content: {row.content}")
    print("Done...")


if __name__ == "__main__":
    # Document Store Config
    dict_faiss_config = {
        'db_path': 'faiss/faiss_dataframe_document_store.db',
        'faiss_index_path': 'faiss/faiss_dataframe_document_store.faiss',
        'faiss_config_path': 'faiss/faiss_dataframe_document_store.json'
    }

    # Sample DataFrame
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
    documents = df[['id', 'content']].to_dict(orient='records')

    # Call Indexing
    create_index(documents, dict_faiss_config)

    # Call Search
    perform_query('fatigue', dict_faiss_config)
    perform_query('fattigue', dict_faiss_config)
