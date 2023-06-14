import os
import pickle
import pandas as pd

saves_ab_len_folder = './chatta/ab_test/context_length/saves/'
saves_ab_ctx_2450_folder = './chatta/ab_test/context_2450/saves/'
saves_ab_ctx_history_folder = './chatta/ab_test/context_history/saves/'


def get_ab_df(saves_folder):
    saves_ab_file_names = [os.path.join(
        saves_folder, name) for name in os.listdir(saves_folder)]
    # use latest save
    if saves_ab_file_names:
        saves_ab_file_names.sort(key=os.path.getctime)
        with open(saves_ab_file_names[-1], 'rb') as f:
            df_ab = pickle.loads(f.read())
    return df_ab
