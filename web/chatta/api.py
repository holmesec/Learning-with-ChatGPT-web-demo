from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
import os
import uuid
import random
import numpy as np
import time
import pickle
import openai
from chatta.ab_test.utils import get_ab_df, saves_ab_len_folder, saves_ab_ctx_2450_folder, saves_ab_ctx_history_folder
from chatta.utils import get_progress
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(
    'sentence-transformers/multi-qa-mpnet-base-dot-v1')
load_dotenv()

bp = Blueprint('api', __name__, url_prefix='/api')

ab_len_test_queue = {}
df_ab_len = get_ab_df(saves_ab_len_folder)

ab_ctx_2450_test_queue = {}
df_ab_ctx_2450 = get_ab_df(saves_ab_ctx_2450_folder)

ab_ctx_history_queue = {}
df_ab_ctx_history = get_ab_df(saves_ab_ctx_history_folder)


@bp.route('/status/<id>', methods=["GET"])
def get_status(id):
    embeddings_path = os.path.join(
        current_app.config['UPLOAD_FOLDER'], 'embeddings', f'{id}.pkl')
    ready = os.path.isfile(embeddings_path)
    if not ready:
        progress = get_progress(id)
        if not progress:
            return jsonify({'error': 'unknown id!'})
        else:
            return jsonify({'is_ready': ready, 'progress': progress})

    return jsonify({'is_ready': ready})


@bp.route('/query', methods=["POST"])
def query():
    data = request.get_json()
    if not "question" in data:
        return "question missing", 400
    if not "id" in data:
        return "id missing",  400

    question = data['question']
    id = data['id']
    file_path = os.path.join(
        current_app.config['UPLOAD_FOLDER'], 'embeddings', f'{id}.pkl')

    if not os.path.isfile(file_path):
        return jsonify({'error': 'unknown id!'})

    embedded_question = embedding_model.encode(question)

    with open(file_path, 'rb') as f:
        context_to_embedding = pickle.loads(f.read())

    best_context = next(iter(context_to_embedding))
    best_score = np.dot(embedded_question, context_to_embedding[best_context])

    for c, e in context_to_embedding.items():
        score = np.dot(embedded_question, e)
        if score > best_score:
            best_context = c
            best_score = score

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "user",
             "content": f'Act as a teacher. A student ask the following question: {question}. Use the following context to answer the question: ```{best_context}```'}
        ]
    )

    answer = completion.choices[0].message.content

    return jsonify({'answer': answer, 'context': best_context})


@bp.route('/ab-test/context-length/fetch', methods=["GET"])
def ab_test_len_fetch():

    # free sample if it has not been labelled 10 min after it was fetched
    df_ab_len.loc[(df_ab_len['choice'].isna()) & (df_ab_len['fetched'] == True) & (
        int(time.time()) - df_ab_len['fetched_time'] > 10*60), 'fetched'] = False

    if len(df_ab_len[df_ab_len['fetched'] == False]) == 0:
        n_unlabelled = len(df_ab_len[df_ab_len["choice"].isna()])
        if n_unlabelled > 0:
            return jsonify({'msg': f'No more data to fetch, but there\'s still {n_unlabelled} unlabelled data-points'})
        else:
            return jsonify({'msg': f'No more data to label!'})

    row = df_ab_len[df_ab_len['fetched'] == False].sample()
    row_index = row.index
    row = row.squeeze()

    question = row['question']
    answer_short = row['answer_short']
    answer_medium = row['answer_medium']
    answer_long = row['answer_long']

    df_ab_len.loc[row_index, 'fetched'] = True
    df_ab_len.loc[row_index, 'fetched_time'] = int(time.time())

    order = random.randint(0, 2)
    id = uuid.uuid4().hex
    ab_len_test_queue[id] = {'order': order, 'row_index': row_index}

    answer_a = answer_short if order == 0 else answer_medium if order == 1 else answer_long
    answer_b = answer_medium if order == 0 else answer_long if order == 1 else answer_short
    answer_c = answer_long if order == 0 else answer_short if order == 1 else answer_medium

    return jsonify({'id': id, 'question': question, 'answer_a': answer_a, 'answer_b': answer_b, 'answer_c': answer_c})


@bp.route('/ab-test/context-length/submit', methods=["POST"])
def ab_test_len_label():
    try:
        data = request.get_json()
        id = data['id']
        choice = data['choice']

        order = ab_len_test_queue[id]['order']
        row_index = ab_len_test_queue[id]['row_index']

        label_list = ['short', 'medium', 'long'] if order == 0 else [
            'medium', 'long', 'short'] if order == 1 else ['long', 'short', 'medium']

        choice_label = label_list[choice]

        df_ab_len.loc[row_index, 'choice'] = choice_label

        n_labelled = len(df_ab_len[~df_ab_len["choice"].isna()])

        if n_labelled > 0 and n_labelled % 10 == 0:
            with open(os.path.join(saves_ab_len_folder, f'{n_labelled}.pkl'), 'wb') as f:
                f.write(pickle.dumps(df_ab_len))

        if len(df_ab_len[df_ab_len["choice"].isna()]) == 0:
            with open(os.path.join(saves_ab_len_folder, f'final.pkl'), 'wb') as f:
                f.write(pickle.dumps(df_ab_len))

        return jsonify({'success': True})

    except:
        return jsonify({'success': False})


@bp.route('/ab-test/context/<name>/fetch', methods=["GET"])
def ab_test_ctx_fetch(name):
    if name not in ['2450', 'history']:
        return jsonify({"msg": "Error: A/B test not found!"})
    df_ab = df_ab_ctx_2450 if name == '2450' else df_ab_ctx_history

    # free sample if it has not been labelled 10 min after it was fetched
    df_ab.loc[(df_ab['choice'].isna()) & (df_ab['fetched'] == True) & (
        int(time.time()) - df_ab['fetched_time'] > 10*60), 'fetched'] = False

    if len(df_ab[df_ab['fetched'] == False]) == 0:
        n_unlabelled = len(df_ab[df_ab["choice"].isna()])
        if n_unlabelled > 0:
            return jsonify({'msg': f'No more data to fetch, but there\'s still {n_unlabelled} unlabelled data-points'})
        else:
            return jsonify({'msg': f'No more data to label!'})

    row = df_ab[df_ab['fetched'] == False].sample()
    row_index = row.index
    row = row.squeeze()

    question = row['question']
    answer = row['answer']
    answer_ctx = row['answer_context']

    df_ab.loc[row_index, 'fetched'] = True
    df_ab.loc[row_index, 'fetched_time'] = int(time.time())

    order = random.randint(0, 1)
    id = uuid.uuid4().hex
    if name == '2450':
        ab_ctx_2450_test_queue[id] = {'order': order, 'row_index': row_index}
    else:
        ab_ctx_history_queue[id] = {'order': order, 'row_index': row_index}

    answer_a = answer if order == 0 else answer_ctx
    answer_b = answer_ctx if order == 0 else answer

    return jsonify({'id': id, 'question': question, 'answer_a': answer_a, 'answer_b': answer_b})


@bp.route('/ab-test/context/<name>/submit', methods=["POST"])
def ab_test_ctx_label(name):
    try:
        if name not in ['2450', 'history']:
            return jsonify({"msg": "Error: a/b test not found!"})
        df_ab = df_ab_ctx_2450 if name == '2450' else df_ab_ctx_history

        data = request.get_json()
        id = data['id']
        choice = data['choice']

        if name == '2450':
            order = ab_ctx_2450_test_queue[id]['order']
            row_index = ab_ctx_2450_test_queue[id]['row_index']
        else:
            order = ab_ctx_history_queue[id]['order']
            row_index = ab_ctx_history_queue[id]['row_index']

        label_list = ['answer', 'answer_context'] if order == 0 else [
            'answer_context', 'answer']

        choice_label = label_list[choice]

        df_ab.loc[row_index, 'choice'] = choice_label

        n_labelled = len(df_ab[~df_ab["choice"].isna()])

        saves_folder = saves_ab_ctx_2450_folder if name == '2450' else saves_ab_ctx_history_folder

        if n_labelled > 0 and n_labelled % 10 == 0:
            with open(os.path.join(saves_folder, f'{n_labelled}.pkl'), 'wb') as f:
                f.write(pickle.dumps(df_ab))

        if len(df_ab[df_ab["choice"].isna()]) == 0:
            with open(os.path.join(saves_folder, f'final.pkl'), 'wb') as f:
                f.write(pickle.dumps(df_ab))

        return jsonify({'success': True})

    except:
        return jsonify({'success': False})
