# Read section.pkl
import pickle
from sentence_transformers import SentenceTransformer
PATH = 'dev\Jason'
with open(PATH + '\sections.pkl', 'rb') as f:
    sections = pickle.load(f)
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

embeddings = model.encode(sections[:1])   # TOO SLOW

with open(PATH + '\section_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)