from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.nodes import EmbeddingRetriever, PromptNode
import os

converter = PDFToTextConverter()
preprocessor = PreProcessor()
db_path = './faiss_document_store.db'
faiss_index_path= './faiss_document_store.json'
faiss_config_path = './faiss_document_config'
document_store = FAISSDocumentStore(sql_url=f"sqlite:///{db_path}")
document_store.save(faiss_index_path, faiss_config_path)

# Add data source
data_path = './Data/journal_llama2.pdf'
indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=converter, name="PDFConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["PDFConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])
indexing_pipeline.run(file_paths=[data_path])

# Add query
document_store = FAISSDocumentStore.load(index_path=faiss_index_path, config_path=faiss_config_path)
retriever = EmbeddingRetriever(document_store = document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
prompt_node = PromptNode(model_name_or_path = "gpt-4",
                         api_key = os.environ.get('CHATGPT_API_KEY'),
                         default_prompt_template = "deepset/question-answering-with-references")

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

question1 = "What is Toxicity?"
result = query_pipeline.run(query = question1)
print("Query Result:", result['results'])

question2 = "What is Bias?"
result = query_pipeline.run(query = question2)
print("Query Result:", result['results'])
