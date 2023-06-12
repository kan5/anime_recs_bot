import asyncio

import aiohttp
import logging
import time
import json
import requests
import os
from pathlib import Path
import traceback
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.joinpath('bot').joinpath('.env'))

data_path = Path(__file__).parent.parent
data_path = data_path.joinpath('data')

token_path = data_path.joinpath("token.json")
mal_access_token = json.load(open(token_path, "r"))



last_mal_api_query = time.time()
rate_limit_api = 1


def update_token(token, CLIENT_ID, CLIENT_SECRET, retry_timeout=1):
    url = 'https://myanimelist.net/v1/oauth2/token'
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': token['refresh_token']
    }
    try:
        response = requests.post(url, data=data)
        if response.status_code in [500, 504] and retry_timeout < 30:
            # This can occur if MAL servers go down or if the page doesnt exist
            raise Exception(f"{response.status_code}")

        return response.json()
    except Exception as e:
        logging.warning(
            f"Received error {str(e)} while accessing {url}. Retrying in {retry_timeout} seconds"
        )
        logging.error(traceback.format_exc())
        time.sleep(retry_timeout)
        return update_token(token, CLIENT_ID, CLIENT_SECRET, retry_timeout=retry_timeout * 2)


def rewrite_token(token, token_path):
    with open(token_path, 'w') as file:
        json.dump(token, file, indent = 4)
        logging.info('Token saved.')


async def call_api(url, retry_timeout=1, max_timeout=100):
    global last_mal_api_query
    global rate_limit_api
    global mal_access_token

    while (time.time() - last_mal_api_query) < rate_limit_api:
        await asyncio.sleep(0.2)

    try:
        async with aiohttp.ClientSession(auto_decompress=False) as session:
            async with session.get(url=url,
                                   headers={"Authorization": f'Bearer {mal_access_token["access_token"]}'}) as response:
                last_mal_web_query = time.time()
                # print(await response.json())
                if response.status in [500, 502, 504, 429, 409]:
                    if retry_timeout < max_timeout:
                    # This can occur if MAL servers go down or if the page doesnt exist
                        raise Exception(f"{response.status}")
                    else:
                        return -1
                if response.status in [403, 404]:
                    return None
                rate_limit_api = 1
                response_out = await response.json()
                # print(response_out)

                if response_out.get('error', '') == 'invalid_token':
                    new_token = update_token(mal_access_token,
                                             CLIENT_ID=os.environ.get('CLIENT_ID'),
                                             CLIENT_SECRET=os.environ.get('CLIENT_SECRET'))
                    print(os.environ.get('CLIENT_ID'))
                    rewrite_token(new_token, token_path)
                    mal_access_token = json.load(open(token_path, "r"))

    except Exception as e:
        logging.warning(
            f"Received error {str(e)} while accessing {url}. Retrying in {retry_timeout} seconds"
        )
        print(traceback.format_exc())
        # await asyncio.sleep(retry_timeout)
        retry_timeout = retry_timeout * 2
        rate_limit_api = retry_timeout
        await asyncio.sleep(retry_timeout)
        return await call_api(url, retry_timeout=retry_timeout)

    return response_out


def parse_json_node(x):
    ls = x["list_status"]
    anime = x["node"]
    # if (ls.get("score", -1) < 8) and (ls.get("status", "") != ):
    #     return None
    entry = {
        # "uid": [x["node"]["id"]],
        "aid": anime.get("id", -1),
        "status": ls.get("status", ""),
        "score": ls.get("score", -1),
        # "num_episodes_watched": [ls.get("num_episodes_watched", -1)],
        # "is_rewatching": [ls.get("is_rewatching", False)],
        # "start_date": [ls.get("start_date", "")],
        # "finish_date": [ls.get("finish_date", "")],
        # "priority": [ls.get("priority", -1)],
        # "num_times_rewatched": [ls.get("num_times_rewatched", -1)],
        # "rewatch_value": [ls.get("rewatch_value", -1)],
        "updated_at": ls.get("updated_at", ""),
        }
    return entry


def process_json(json):
    entries = [parse_json_node(x) for x in json["data"]]
    entries = [i for i in entries if i]
    if entries:
        return entries
    else:
        return []


async def get_user_anime_list(username):
    anime_lists = []
    more_pages = True
    url = f"https://api.myanimelist.net/v2/users/{username}/animelist?limit=1000&fields=list_status&nsfw=true"
    while more_pages:
        response = await call_api(url)
        if not response:
            return -1

        json = response
        anime_lists += process_json(json)
        more_pages = "next" in json["paging"]
        if more_pages:
            url = json["paging"]["next"]

    if anime_lists:
        out_lists = {i: [] for i in anime_lists[0].keys()}
        for i in anime_lists:
            for key, value in i.items():
                out_lists[key].append(value)
        user_anime_list = {"anime_list": out_lists}
        user_anime_list["username"] = username

        return out_lists
    else:
        return None