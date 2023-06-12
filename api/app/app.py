import time
from typing_extensions import Annotated
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import json

from app_utils import get_user_anime_list
from model import full_pipeline


app = FastAPI()


@app.get("/")
async def hello_world(request: Request):
    # return json.dumps({'error': 'bobabuba'})
    username = (await request.json()).get('username')
    user_anilist = await get_user_anime_list(username)
    if user_anilist == -1:
        return json.dumps({'error': 'mal server sucks'})
    elif not user_anilist:
        return json.dumps({'error': 'no matching titles'})
    a = json.dumps(full_pipeline(anime_ids=user_anilist['aid'],
                         scores=user_anilist['score'],
                         status=user_anilist['status'],
                         ts=user_anilist['updated_at'], is_seq=True))
    return a

if __name__ == '__main__':
    uvicorn.run("app:app", port=1111, host='127.0.0.1')