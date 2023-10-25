
# Anime recommender telegram bot

This is a telegram bot, that can recommend anime based on your watched titles retrived from [MyAnimeList](https://myanimelist.net) profile. Also it has 2 characters: anime girls that send you voice messages.

There using BERT4rec to recommend and SHAP to explain results.

# Additional materials
Youtube video demonstartion - https://youtu.be/FTQkJ_1QYe0
Thesis in pdf - https://drive.google.com/file/d/1RAlhXHM5VXTxHMHDxKKXvmvBVhBcOe4J/view?usp=sharing
Miro dialog tree - https://miro.com/app/board/uXjVMS2CQnA=/?share_link_id=443006680615

## Stack

- Dataset - [Kaggle](https://www.kaggle.com/datasets/azathoth42/myanimelist)
- Library for modeling - [RecBole](https://github.com/RUCAIBox/RecBole)
- Library for explaining - [SHAP](https://shap.readthedocs.io/en/latest/index.html)
- Notebook - [Kaggle](https://www.kaggle.com/code/payngay/preparing-data-for-recbole)
- telegram bot - aiogram
- voice generator - voicevox

## Launch settings

1. install packages for python 3.7
2. add token.json from myanimelist app/data/token.json
3. add .env to bot/.env with  
API_TOKEN=from telegram bot  
CLIENT_ID=from mal_api  
CLIENT_SECRET=from mal_api  
4. install and launch mongodb
5. launch api/app.py
6. launch bot/bot.py
