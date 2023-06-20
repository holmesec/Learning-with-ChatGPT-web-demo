import re
import os
import pickle
import pypdfium2 as pdfium
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(
    'sentence-transformers/multi-qa-distilbert-cos-v1')

UPLOAD_FOLDER = 'chatta/uploads'


id_to_progress = {}


def get_progress(id):
    # print(id)
    # print(id_to_progress[id])
    return id_to_progress.get(id)


def extract_sentences_from_pdf(id):
    pdf = pdfium.PdfDocument(os.path.join(UPLOAD_FOLDER, 'pdf', f'{id}.pdf'))
    text_all = ""
    for page in pdf:
        textpage = page.get_textpage()
        text_all += " ".join(textpage.get_text_range().splitlines())

    text_split = list(re.split('(\. |\? |\! )', text_all))
    text_sentences = []
    i = 0
    n = len(text_split)
    while i < n-1:
        text_sentences.append(text_split[i]+text_split[i+1][:-1])
        i += 2
    if i == n-1:
        text_sentences.append(text_split[i])
    text_sentences = [s for s in text_sentences if len(s) > 10]

    return text_sentences


def generate_contexts_from_pdf_text(sentences):
    contexts = []
    seg = ""
    threshold = 3000  # chars
    for s in sentences:
        if len(seg)+len(s) > threshold:
            contexts.append(seg.strip())
            seg = s
        else:
            seg += f' {s}'
    return contexts


def generate_embeddings_from_contexts(contexts, id):
    embeddings = {}
    n = len(contexts)
    id_to_progress[id] = f'Generating embeddings: 0/{n}'
    for i, c in enumerate(contexts):
        embeddings[c] = embedding_model.encode(c)
        id_to_progress[id] = f'Generating embeddings: {i+1}/{n}'
    return embeddings


def process_pdf(id):
    id_to_progress[id] = 'Extracting sentences...'
    sentences = extract_sentences_from_pdf(id)

    id_to_progress[id] = 'Generating contexts...'
    contexts = generate_contexts_from_pdf_text(sentences)

    id_to_progress[id] = f'Generating embeddings'
    embeddings = generate_embeddings_from_contexts(contexts, id)

    id_to_progress[id] = 'Saving embeddings contexts...'

    with open(os.path.join(UPLOAD_FOLDER, 'embeddings', f'{id}.pkl'), 'wb') as f:
        f.write(pickle.dumps(embeddings))

    del id_to_progress[id]
