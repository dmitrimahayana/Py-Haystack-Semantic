import pandas as pd
from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PreProcessor
from haystack.nodes import EmbeddingRetriever, PromptNode
import os

preprocessor = PreProcessor()

db_path = 'faiss/faiss_dataframe_document_store.db'
faiss_index_path = "faiss/faiss_dataframe_document_store.faiss"
faiss_config_path = "faiss/faiss_dataframe_document_store.json"
if os.path.exists(db_path):
    os.remove(db_path)
if os.path.exists(faiss_index_path):
    os.remove(faiss_index_path)
if os.path.exists(faiss_config_path):
    os.remove(faiss_config_path)
document_store = FAISSDocumentStore(sql_url=f"sqlite:///{db_path}")

retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Sample DataFrame
data = {
    'id': [1, 2, 3, 4, 5],
    'url': ['url1', 'url2', 'url3', 'url4', 'url5'],
    'title': ['title1', 'title2', 'title3', 'title4', 'title5'],
    'desc': ['description1 fatigue', 'description2', 'fatigue description3', 'description4', 'no fatigue'],
    'location': ['loc1', 'loc2', 'loc3', 'loc4', 'loc5']
}
df = pd.DataFrame(data)
df['content'] = df['title'] + ' ' + df['desc']
documents = df[['id', 'content']].to_dict(orient='records')

# Update Doc Embedding
document_store.write_documents(documents)
document_store.update_embeddings(retriever)
document_store.save(faiss_index_path)

# Reload Document Indexing
document_store = FAISSDocumentStore(
    faiss_index_path=faiss_index_path,
    faiss_config_path=faiss_config_path)

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
results = query_pipeline.run(query="fatigue", params={"Retriever": {"top_k": 3}})
for row in results['documents']:
    print(f"ID: {row.id}, Content: {row.content}")
print("Done...")