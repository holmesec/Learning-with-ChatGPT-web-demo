import os
import pickle
import pandas as pd

saves_ab_len_folder = './chatta/ab_test/context_length/saves/'


def get_ab_len_df():
    saves_ab_len_file_names = [os.path.join(
        saves_ab_len_folder, name) for name in os.listdir(saves_ab_len_folder) if name != '.gitkeep']

    # use latest save if any
    if saves_ab_len_file_names:
        saves_ab_len_file_names.sort(key=os.path.getctime)
        with open(saves_ab_len_file_names[-1], 'rb') as f:
            df_ab_len = pickle.loads(f.read())
    else:
        with open('./chatta/ab_test/context_length/df_ab_original.pkl', 'rb') as f:
            df_ab_len = pickle.loads(f.read())
    return df_ab_len
