from flask import Blueprint, render_template, request, jsonify
import os

import pypdfium2 as pdfium
from docarray import Document, DocumentArray
from transformers import pipeline

UPLOAD_FOLDER = 'chatta/uploads'

bp = Blueprint('api', __name__, url_prefix='/api')


@bp.route('/query', methods=["POST"])
def query():
    data = request.get_json()
    if not "question" in data:
        return "question missing", 400
    if not "file_id" in data:
        return "file_id missing",  400

    question = data['question']
    file_id = data['file_id']
    file_path = os.path.join(UPLOAD_FOLDER, f'{file_id}.pdf')

    pdf = pdfium.PdfDocument(file_path)
    page = pdf[0]
    textpage = page.get_textpage()
    text_all = textpage.get_text_range()
    text_all = " ".join(text_all.splitlines())
    text_segments = list(filter(None, text_all.split('.')))
    docs = DocumentArray(Document(text = s) for s in text_segments)
    docs.apply(lambda doc: doc.embed_feature_hashing())
    query = (Document(text=question).embed_feature_hashing().match(docs, limit=5, exclude_self=True, metric="jaccard", use_scipy=True))
    oracle = pipeline(model="deepset/roberta-base-squad2")
    answer = oracle(question=question, context=" ".join(query.matches[:,['text']]))
    return jsonify(answer)
