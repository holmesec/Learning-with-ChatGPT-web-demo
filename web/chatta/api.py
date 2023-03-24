from flask import Blueprint, render_template, request, jsonify, current_app
import openai
import os
import pypdfium2 as pdfium
from docarray import Document, DocumentArray
from transformers import pipeline

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
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f'{file_id}.pdf')

    pdf = pdfium.PdfDocument(file_path)
    text_all = ""
    for page in pdf:
        textpage = page.get_textpage()
        text_all += " ".join(textpage.get_text_range().splitlines())
    text_segments = list(filter(None, text_all.split('.')))
    docs = DocumentArray(Document(text = s) for s in text_segments)
    docs.apply(lambda doc: doc.embed_feature_hashing())
    query = (Document(text=question).embed_feature_hashing().match(docs, limit=50, exclude_self=True, metric="jaccard", use_scipy=True))

    openai.api_key = current_app.config['OPENAI_API_KEY']
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f'Act as a teacher. A student ask the following question: {question}. Use the following context to answer the question: {" ".join(query.matches[:,["text"]])}'}
    ]
    )

    answer=completion.choices[0].message.content

    return jsonify({'answer':answer})
