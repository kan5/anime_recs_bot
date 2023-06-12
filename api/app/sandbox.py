# import pandas as pd
# from app_utils import data_path
#
#
# anime_df = pd.read_json(data_path.joinpath('anime_jikan.jsonl'),
#                         nrows=100,
#                         lines=True)
# anime_df['image'] = anime_df.images.map(lambda x: x.get('jpg').get('large_image_url'))
# anime_df.genres = anime_df.genres.map(lambda x: [i.get('name') for i in x])
# anime_df.themes = anime_df.themes.map(lambda x: [i.get('name') for i in x])
#
# anime_df = anime_df[['mal_id', 'url', 'image', 'title', 'score', 'synopsis', 'year', 'genres', 'themes']]
# anime_df.rename(columns={'mal_id': 'anime_id'}, inplace=True)
# anime_df.set_index('anime_id', inplace=True)
# print(anime_df)

print([0 for i in range(1)])