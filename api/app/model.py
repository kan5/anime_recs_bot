import gc

from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer
from logging import getLogger
from recbole.data.interaction import Interaction
import torch
import pandas as pd
import numpy as np
import os
import gc
# from sklearn.model_selection import train_test_split
import pickle
import shap
import io
# from PIL import Image
import matplotlib.pyplot as plt
import base64

from app_utils import data_path
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,3)

model_name = 'BERT4Rec'

sequntal_parameter_dict = {
    #     'nproc': 2,
    #     'gpu_id': '0',
    #     'worker': 2,
    'seed': 2020,
    # 'data_path': output_path,
    'USER_ID_FIELD': 'username',
    'ITEM_ID_FIELD': 'anime_id',
    #     'RATING_FIELD': 'my_score',
    'TIME_FIELD': 'my_last_updated',
    'load_col': {
        'inter': ['anime_id',
                  'username',
                  'my_last_updated',
                  #                   'my_score',
                  #                   'label',
                  ],
    },
    'MAX_ITEM_LIST_LENGTH': 50,
    #     'LABEL_FIELD': 'label',
    "train_neg_sample_args": None,

    #     'user_inter_num_interval': "[3,100)",
    #     'item_inter_num_interval': "[50,inf)",
    "show_progress": False,
    "train_batch_size": 7000,
    "learning_rate": 0.01,
    'epochs': 10,
    'loss_type': 'CE',
    'eval_args': {
        'split': {'RS': [1, 1, 1]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'},
    'metrics': ['MRR', 'NDCG', 'MAP'],
    'topk': 5,
    'valid_metric': 'NDCG@5',
}
max_len = sequntal_parameter_dict['MAX_ITEM_LIST_LENGTH']

config = Config(model=model_name, config_dict=sequntal_parameter_dict,
                config_file_list=[data_path.joinpath('config.yaml')],
                )

# init random seed
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()

# write config info into log
logger.info(config)

# dataset creating and filtering
with open(data_path.joinpath('dataset.pickle'), 'rb') as f:
    dataset = pickle.load(f)
logger.info(dataset)

train_data, valid_data, test_data = data_preparation(config, dataset)

del valid_data
del test_data
gc.collect()

# model loading and initialization
model = BERT4Rec(config, train_data.dataset).to(config['device'])

checkpoint = torch.load(data_path.joinpath('model.pth'),
                        map_location=torch.device('cpu'))  # *.pth file is used for load saved model.
model.load_state_dict(checkpoint['state_dict'])

anime_df = pd.read_json(data_path.joinpath('anime_jikan.jsonl'), lines=True)
anime_df['image'] = anime_df.images.map(lambda x: x.get('jpg').get('large_image_url'))
anime_df.genres = anime_df.genres.map(lambda x: [i.get('name') for i in x])
anime_df.themes = anime_df.themes.map(lambda x: [i.get('name') for i in x])
anime_df = anime_df[['mal_id', 'url', 'image', 'title', 'score', 'synopsis', 'year', 'genres', 'themes']]
anime_df.set_index('mal_id', inplace=True)


def token2title(x: int, els=None):
    try:
        return anime_df['title'].loc[x]
    except ValueError:
        return els


def token2field(x: int, field: str, els=None):
    try:
        return anime_df[field].loc[x]
    except:
        return els

def model_predict(items, len_items):
    input_inter = Interaction({
        'user_id': torch.tensor([1]),
        'anime_id_list': torch.tensor([items + [0 for _ in range(max_len - len(items))]]),
        'item_length': torch.tensor([len_items]),
    })
    with torch.no_grad():
        scores = model.full_sort_predict(input_inter)[0].numpy()

    return (scores)


def item2token(x: int, els=None):
    try:
        return int(dataset.id2token('anime_id', x))
    except ValueError:
        return els


def token2item(x: int, els=None):
    try:
        return dataset.token2id('anime_id', str(x))
    except ValueError:
        return els


def top_k(seq, watched, k=5, is_seq=True):
    new_watched = watched + [0]
    new_seq = seq.copy()
    figures = []
    #      new_seq = [token2item(str(i)) for i in seq if i != '[PAD]']
    print('is_seq', is_seq)
    if is_seq:
        for _ in range(k):
            preds = model_predict(new_seq, len(new_seq))
            sorted_preds = np.argsort(preds)
            max_pred = sorted_preds[-1]
            v = -1
            while (max_pred in new_watched) or (not token2field(item2token(max_pred), 'title')):
                v -= 1
                max_pred = sorted_preds[v]
                # print(token2field(item2token(max_pred), 'title'), item2token(max_pred), max_pred)
            #             a = build_plot(new_seq, max_pred)
            #             im = Image.open(a)
            #             im.show()
            #             a.close()
            # print(max_pred, sorted_preds[max_pred])
            # print('.')
            figures.append(build_plot(new_seq, max_pred))
            new_seq.append(max_pred)
            new_watched.append(max_pred)
        max_ids = new_seq[-k:]
    else:
        preds = model_predict(new_seq[-max_len:], len(new_seq[-max_len:]))
        max_ids = np.argsort(preds)
        max_ids = max_ids[np.where(np.logical_not(np.isin(max_ids, new_watched)))][-k:]
        for i in max_ids:
            figures.insert(0, build_plot(new_seq, i))

    max_ids = max_ids[::-1]
    max_aids = [item2token(i) for i in max_ids]
    # print(max_aids)
    return [{'aid': i, 'explain_image': j} for i, j in zip(max_aids, figures)]


def f(sequences):
    #     print(sequences)
    out = []
    proc_sequences = []
    for seq in sequences:
        seq = seq.replace('...', '')
        seq = seq.split()
        seq = list(map(int, seq))
        proc_sequences.append(seq)
    for seq in proc_sequences:
        len_seq = len(seq)
        if len(seq) == 0:
            seq = [0]
            len_seq = 0
        out.append(model_predict(seq, len_seq))
    #     print(np.array(out))
    return np.array(out)


def build_plot(seq, iid):
    masker = shap.maskers.Text(r"\W")
    explainer = shap.Explainer(f, masker)

    shap_values = explainer([" ".join(map(str, seq))])
    preds = model_predict(seq, len(seq))
    #     max_idx = preds.argmax(axis=0)
    max_idx = iid

    shap_values.feature_names = [list(map(
        lambda x: token2field(item2token(int(x)), 'title'),
        shap_values.feature_names[0]))]
    #     print(shap_values.feature_names)
    # shap.plots.text(shap_values[:,:,max_idx], display=False)
    shap.plots.bar(shap_values[:,:,max_idx].mean(0),
                   clustering=None,
                   clustering_cutoff=0,
                   # plot_cmap=["#335BFF", "#FF5733"],
                   show = False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.clf()
    buf.seek(0)
    a = base64.encodebytes(buf.getvalue()).decode('utf-8')
    #     im = Image.open(buf)
    #     im.show()
    buf.close()
    #     print(buf)
    #     display(Image(data=a))
    return a


def get_shit_about_anime(anime_id: int):
    pass


def full_pipeline(anime_ids, scores, status, ts, is_seq=True, k=5):
    needed_idxs = np.argsort(ts)

    input_aids = []
    input_iids = []
    watched_aids = []
    watched_iids = []
    for i in needed_idxs:
        iid = token2item(anime_ids[i])
        if iid:
            watched_aids.append(anime_ids[i])
            watched_iids.append(iid)
            if (scores[i] >= 8) and (status[i] in ['completed']):
                input_aids.append(anime_ids[i])
                input_iids.append(iid)
        input_iids = input_iids[-max_len+k-1:]
    if not input_iids:
        return {'error': 'no matching titles'}

    if len(input_iids) + k - 1 > max_len:
        is_seq = False
    recommended = top_k(input_iids, watched_iids, k=k, is_seq=is_seq)
    # print(len(recommended))
    for dic in recommended:
        aid = dic.get('aid')
        # print(type(token2field(aid, 'title')), token2field(aid, 'title'))
        dic.update({
            'score': token2field(aid, 'score'),
            'url': token2field(aid, 'url'),
            'image': token2field(aid, 'image'),
            'title': token2field(aid, 'title'),
            'synopsis': token2field(aid, 'synopsis')[:token2field(aid, 'synopsis').rfind('\n')] if token2field(aid, 'synopsis') else None,
            'year': str(token2field(aid, 'year'))[:str(token2field(aid, 'year')).rfind('.')],
            'genres': token2field(aid, 'genres'),
            'themes': token2field(aid, 'themes'),
        })
        dic['aid'] = int(aid)
        dic['score'] = float(dic['score']) if dic['score'] else dic['score']
        # for i, j in dic.items():
        #     print(i, type(j))
    return {'data': recommended}


# dic = {'aid': [47, 22789, 50854, 889, 31043, 31478, 39523, 1, 42310, 35849, 1535, 39792, 37349, 43692, 268, 245, 39030, 1592, 39468, 185, 49220, 14719, 30831, 43, 17265, 32526, 50709, 9693, 39535, 30, 2966, 50739, 30015, 37450, 48736, 50265, 22319, 42249, 33352, 50346, 23283], 'status': ['completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'dropped', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'watching', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed', 'completed'], 'score': [6, 8, 7, 8, 9, 6, 7, 8, 7, 8, 4, 10, 6, 7, 8, 9, 7, 7, 9, 8, 8, 7, 8, 6, 6, 7, 7, 7, 9, 6, 8, 8, 10, 8, 7, 7, 7, 7, 10, 8, 8], 'updated_at': ['2023-03-07T18:57:42+00:00', '2023-02-02T17:06:33+00:00', '2023-04-26T18:26:41+00:00', '2023-02-02T16:53:33+00:00', '2023-03-07T18:58:38+00:00', '2023-02-02T16:59:07+00:00', '2023-03-07T18:55:32+00:00', '2023-02-02T17:09:59+00:00', '2023-03-07T18:54:10+00:00', '2023-02-02T17:11:00+00:00', '2023-03-07T19:04:25+00:00', '2023-02-02T17:02:49+00:00', '2023-03-07T18:56:14+00:00', '2023-02-02T17:05:07+00:00', '2023-03-07T18:57:14+00:00', '2023-02-02T17:09:26+00:00', '2023-02-02T17:05:53+00:00', '2023-02-02T16:58:15+00:00', '2023-02-02T17:02:18+00:00', '2023-02-02T16:46:26+00:00', '2023-03-07T19:40:30+00:00', '2023-03-07T19:05:16+00:00', '2023-03-07T18:54:59+00:00', '2023-03-07T19:06:53+00:00', '2023-02-02T17:04:15+00:00', '2023-02-02T17:01:25+00:00', '2023-03-07T18:53:37+00:00', '2023-02-02T17:00:09+00:00', '2023-02-02T16:49:20+00:00', '2023-02-02T17:08:54+00:00', '2023-03-07T19:02:47+00:00', '2023-04-26T18:26:02+00:00', '2023-02-02T16:48:26+00:00', '2023-02-02T17:00:50+00:00', '2023-02-02T16:54:53+00:00', '2023-02-02T17:10:23+00:00', '2023-04-26T20:18:16+00:00', '2023-02-02T16:56:54+00:00', '2023-03-07T19:02:18+00:00', '2023-03-07T19:02:29+00:00', '2023-02-02T17:08:13+00:00']}
# a = full_pipeline(anime_ids=dic['aid'],
#               scores=dic['score'],
#               status=dic['status'],
#               ts=dic['updated_at'], is_seq=True)
# print(a)
