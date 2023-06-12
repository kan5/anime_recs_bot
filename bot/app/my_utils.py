import asyncio

import aiofiles
import aiohttp
import lxml.html
import logging
from pathlib import Path
import time
from typing import Callable
import json
import requests
import os

from aiogram.types import CallbackQuery
from aiogram.types.chat import ChatActions
from aiogram.types.message import Message
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient()
db = client['anime_recs_bot']
history = db['messages']

data_path = Path(__file__).parent.parent
data_path = data_path.joinpath('data')

logging.basicConfig(level=logging.DEBUG, filename=data_path.joinpath("bot_log.log"), filemode="w")

last_mal_web_query = time.time()
rate_limit_web = 1 # time between queries

last_mal_api_query = time.time()
rate_limit_api = 1

async def record_answer(answer):
    record_dict = {}
    record_dict['msg'] = dict(answer)
    await history.insert_one(record_dict)


def add_event(func):
    async def fn(*args, **kwargs):
        record_dict = {}
        if type(args[0]) == Message:
            record_dict['msg'] = dict(args[0])
        elif type(args[0]) == CallbackQuery:
            record_dict['msg'] = dict(args[0])
        else:
            raise Exception()
        record_dict['state'] = await kwargs['state'].get_state()
        record_dict['user_data'] = await kwargs['state'].get_data()
        if 'data' in record_dict['user_data']:
            del record_dict['user_data']['data']

        await history.insert_one(record_dict)
        new_kwargs = {key: value for key, value in kwargs.items() if key in ['state']}
        return await func(*args, **new_kwargs)
    return fn

async def set_typing_status(message: Message,
                            pre_delay: float = 1,
                            typing_time: float = 5,
                            action_type: ChatActions = ChatActions.TYPING):

    record_dict = {}
    record_dict['msg'] = {'chat': dict(message)['chat']}
    record_dict['msg'].update({'from': {'is_bot': True},
                               'action_type': str(action_type),
                               'pre_delay': pre_delay,
                               'typing_time': typing_time,
                               })
    await history.insert_one(record_dict)

    await asyncio.sleep(pre_delay)
    await send_action(message, action=action_type)
    time_conrol = time.time()
    is_executed = False
    out = None
    while (time.time() - time_conrol) < typing_time:
        loop_start = time.time()
        delay_calc = 5 - (time.time() - loop_start)
        global_reminder = typing_time - (time.time() - time_conrol)
        delay = min(global_reminder, delay_calc)
        await asyncio.sleep(delay)
        await send_action(message, action=action_type)

    return out


async def send_voice(message: Message, filename: str, text=''):
    fin_path =data_path.joinpath('audio').joinpath(f'{filename}.opus')
    if os.path.isfile(fin_path):
        async with aiofiles.open(fin_path, mode='rb') as f:
            answer = await message.answer_voice(f, protect_content=True, caption=text)
    else:
        async with aiofiles.open(data_path.joinpath('audio').joinpath('some.opus'), mode='rb') as f:
            answer = await message.answer_voice(f, protect_content=True, caption=text)

    await record_answer(answer)

    return answer


async def send_text(message: Message, **kwargs):
    answer = await message.answer(**kwargs)
    await record_answer(answer)
    return answer

async def send_sticker(message: Message, **kwargs):
    answer = await message.answer_sticker(**kwargs)
    await record_answer(answer)
    return answer

async def send_photo(message: Message, **kwargs):
    answer = await message.answer_photo(**kwargs)
    await record_answer(answer)
    return answer

async def send_action(message: Message, **kwargs):
    answer = await message.answer_chat_action(**kwargs)
    # record_dict = {}
    # record_dict['msg'] = {'chat': dict(message.get_current())['chat']}
    # record_dict['msg'].update(kwargs)
    # await history.insert_one(record_dict)
    return answer

async def call_web(url, retry_timeout=1, max_timeout=3):
    global last_mal_web_query
    global rate_limit_web

    while (time.time() - last_mal_web_query) < rate_limit_web:
        await asyncio.sleep(0.2)

    try:
        async with aiohttp.ClientSession(auto_decompress=False) as session:
            async with session.get(url=url) as response:
                last_mal_web_query = time.time()
                if response.status in [500, 502, 504, 429, 409]:
                    if retry_timeout < max_timeout:
                        raise Exception(f"call_web {response.status}")
                    else:
                        return -1
                if response.status in [403, 404]:
                    return None
                response_out = await response.content.read()
                rate_limit_web = 1
    except Exception as e:
        logging.warning(
            f"Received error {str(e)} while accessing {url}. Retrying in {retry_timeout} seconds"
        )
        # await asyncio.sleep(retry_timeout)
        retry_timeout = retry_timeout * 2
        rate_limit_web = retry_timeout
        return await call_web(url, retry_timeout)

    return response_out


async def get_user_info(username):
    url = f"https://myanimelist.net/profile/{username}"
    response = await call_web(url)
    if not response:
        return None
    elif response == -1:
        return -1
    tree = lxml.html.fromstring(response)
    # response = requests.get('https://myanimelist.net/profile/xandra98', stream=True)
    # response.raw.decode_content = True
    # tree = lxml.html.parse(response.read())
    try:
        user_dict = {i.xpath('./span/text()')[0]: i.xpath('./span/text()')[1] \
                     for i in
                     tree.xpath('//ul[@class="user-status border-top pb8 mb4"]')[0].xpath('./li[@class="clearfix"]')}
        user_dict.update({i.xpath('./span/text()')[0]: i.xpath('./span/text()')[1] \
                          for i in
                          tree.xpath('//ul[@class="stats-data fl-r"]')[0].xpath('./li[@class="clearfix mb12"]')})
        user_dict['username'] = username
        user_dict['image'] = tree.xpath('.//*[@id="content"]/div/div[1]/div/div[1]/img/@data-src')
        if user_dict['image']:
            user_dict['image'] = user_dict['image'][0]

        # if user_dict['image']:
        #     async with aiohttp.ClientSession() as session:
        #         url = user_dict['image'][0]
        #         async with session.get(url) as resp:
        #             if resp.status == 200:
        #                 user_dict['image'] = await resp.read()
        # else:
        #     user_dict['image'] = None

        return user_dict
    except:
        return -1


async def call_model(url, data, loop_count=1, max_timeout=10):
    try:
        async with aiohttp.ClientSession(auto_decompress=False) as session:
            async with session.get(url=url,
                                   json=data) as response:
                last_mal_web_query = time.time()
                if response.status in [500, 502, 504, 429, 409]:
                    return -1
                if response.status in [403, 404]:
                    return None
                return json.loads(await response.json())
    except Exception as e:
        logging.warning(
            f"Received error {str(e)} while accessing {url}."
        )
        await asyncio.sleep(1)
        return await call_model(url, data, loop_count+1)

    # return json.loads(response_out)


# async def fastapi_req(data):
#     return await call_api('http://127.0.0.1:1111/')