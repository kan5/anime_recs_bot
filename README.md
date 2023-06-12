
# Anime recommender telegram bot

This is a telegram bot, that can recommend anime based on your watched titles retrived from [MyAnimeList](https://myanimelist.net) profile. Also it has 2 characters: anime girls that send you voice messages.

There using BERT4rec to recommend and SHAP to explain results.

# Demo
[Youtube video](https://youtu.be/FTQkJ_1QYe0)

## Stack

- Dataset - [Kaggle](https://www.kaggle.com/datasets/azathoth42/myanimelist)
- Library for modeling - [RecBole](https://github.com/RUCAIBox/RecBole)
- Library for explaining - [SHAP](https://shap.readthedocs.io/en/latest/index.html)
- Notebook - [Kaggle](https://www.kaggle.com/code/payngay/preparing-data-for-recbole)
- telegram bot - aiogram
- voice generator - voicevox