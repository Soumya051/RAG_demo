from document_parser import DocumentParser
from utils import Utils
import numpy as np
import pandas as pd

class MockVectorDatabase:
  def __init__(self, doc_parser_model_name, col_length = 384):
    self.doc_parser = DocumentParser(model_name = doc_parser_model_name)
    self.datastore = pd.DataFrame(columns = list(range(col_length)))
    self.metadata = []
    self.count = 0

  def insert_entry(self, filepath: str):
    data = self.doc_parser.read_file(filepath)
    self.datastore.loc[self.count] = data['embedding']
    self.metadata.append({'text': data['text'], 'magnitude': Utils.magnitude(data['embedding'])})
    self.count+=1

  def get_relevant_contents(self, query: str, top_n: int = 2):
    query_embedding = self.doc_parser.get_embedding(data = query)
    q_magnitude = Utils.magnitude(query_embedding)
    cosine_similarity = [0]*self.count
    for i in range(self.count):
      content_embedding = self.datastore.loc[i].values
      dot_p = np.dot(query_embedding, content_embedding)
      cosine_similarity[i] = dot_p/(q_magnitude*self.metadata[i]['magnitude'])
    order = np.argsort(cosine_similarity)[::-1]
    relevant_contents = []
    for i in order[:top_n]:
      relevant_contents.append({'score': cosine_similarity[i], 'content': self.metadata[i]})
    
    return relevant_contents