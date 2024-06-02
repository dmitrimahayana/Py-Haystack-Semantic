from haystack.nodes import BaseComponent


class MetadataFilter(BaseComponent):
    outgoing_edges = 1  # This component has one outgoing edge

    def __init__(self):
        super().__init__()
        self.basic_filters = None

    def run(self, documents, filter, top_k=100):
        self.basic_filters = filter
        filtered_documents = [doc for doc in documents if self._matches_filters(doc)]
        sorted_documents = sorted(filtered_documents, key=lambda doc: doc.score, reverse=True)
        sorted_documents = sorted_documents[:top_k]
        return {"documents": sorted_documents}, "output_1"

    def run_batch(self, documents):
        return self.run(documents)

    def _matches_filters(self, document):
        if self.basic_filters is not None and len(self.basic_filters) > 0:
            if 'stage' in self.basic_filters:
                if document.meta.get('stage').lower() == self.basic_filters['stage'.lower()] and self.basic_filters['stage'] is not None:
                    document.score = min(document.score + 0.8, 1)
            if 'type' in self.basic_filters:
                if document.meta.get('type').lower() == self.basic_filters['type'].lower() and self.basic_filters['type'] is not None:
                    document.score = min(document.score + 0.8, 1)
            if 'attribute' in self.basic_filters:
                if document.meta.get('attribute').lower() == self.basic_filters['attribute'].lower() and self.basic_filters['attribute'] is not None:
                    document.score = min(document.score + 0.8, 1)
        return True