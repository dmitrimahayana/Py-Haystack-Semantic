from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack import Document
import os

converter = PDFToTextConverter()
preprocessor = PreProcessor()

db_path = './faiss_document_store.db'
faiss_index_path = "./faiss_document_store.faiss"
faiss_config_path = "./faiss_document_store.json"
if os.path.exists(db_path):
  os.remove(db_path)
if os.path.exists(faiss_index_path):
  os.remove(faiss_index_path)
if os.path.exists(faiss_config_path):
  os.remove(faiss_config_path)
document_store = FAISSDocumentStore(sql_url=f"sqlite:///{db_path}")

retriever = EmbeddingRetriever(document_store = document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

prompt_node = PromptNode(model_name_or_path = "gpt-4",
                         api_key = os.environ.get('CHATGPT_API_KEY'),
                         default_prompt_template = "deepset/question-answering-with-references")

# Create Indexing
data_path1 = './Data/journal_llama2.pdf'
data_path2 = './Data/Test.pdf'
indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=converter, name="PDFConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["PDFConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])
indexing_pipeline.run(file_paths=[data_path1, data_path2])

# Update Doc Embedding
document_store.update_embeddings(retriever)
document_store.save("faiss_document_store.faiss")

# Reload Document Indexing
document_store = FAISSDocumentStore(
    faiss_index_path="faiss_document_store.faiss",
    faiss_config_path="faiss_document_store.json")

# print(document_store.get_all_documents())

# Load and Query Index
query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

question = "who is dmitri?"
result = query_pipeline.run(query = question)
print("\nQuery Result:", result['results'])

question = "what is toxicity?"
result = query_pipeline.run(query = question)
print("\nQuery Result:", result['results'])

question = "What is bias?"
result = query_pipeline.run(query = question)
print("\nQuery Result:", result['results'])