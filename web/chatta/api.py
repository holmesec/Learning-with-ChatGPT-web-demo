from flask import Blueprint, request, jsonify, current_app
import os
import uuid
import random
import pandas as pd
import time
import pickle
from chatta.ab_test.utils import get_ab_len_df, saves_ab_len_folder

bp = Blueprint('api', __name__, url_prefix='/api')

ab_len_test_queue = {}
df_ab_len = get_ab_len_df()


@bp.route('/query', methods=["POST"])
def query():
    return jsonify({'answer': '42'})


@bp.route('/ab-test/contex-length/submit', methods=["POST"])
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


@bp.route('/ab-test/contex-length/fetch', methods=["GET"])
def ab_test_ctx_fetch():

    # free sample if it has not been labelled 10 min after it was fetched
    df_ab_len.loc[(df_ab_len['choice'].isna()) & (df_ab_len['fetched'] == True) & (
        int(time.time()) - df_ab_len['fetched_time'] > 10*60), 'fetched'] = False

    if len(df_ab_len[df_ab_len['fetched'] == False]) == 0:
        n_unlabelled = len(df_ab_len[df_ab_len["choice"].isna()])
        if n_unlabelled > 0:
            return jsonify({'msg': f'No more data to fetch, but there\'s still {n_unlabelled}unlabelled data-points'})
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
