import time

from aiogram import Bot, types, executor
from aiogram.contrib.fsm_storage.mongo import MongoStorage
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils.helper import Helper, HelperMode, ListItem
from aiogram.types.chat import ChatActions
import asyncio
import json
import os
from app.handlers_reg import *
from app.my_utils import data_path
import logging
from dotenv import load_dotenv

load_dotenv('.env')



# @dp.message_handler()
# async def echo(message: types.Message):
#     time_conrol = time.time()
#     # await set_typing_status(message=message, pre_delay=1.5, typing_time=10, action_type=ChatActions.RECORD_VOICE)
#     # answ = await call_api('http://127.0.0.1:1111/')
#     # if 'image' in answ:
#     #     answ['image'] = 0
#     # res = 'answer'
#     # res = answ.decode("utf-8") + ' ' + str(round(time.time() - time_conrol, 2))
#     # print(json.dumps(await get_user_anime_list('jiga-jiga'), indent=4))
#     # print(await get_user_anime_list('jiga-jiga'))
#     await message.answer(
#         # await get_user_anime_list('jiga-jiga')
#         f'got {message.text}'
#     )
#     a = await call_model('http://127.0.0.1:1111/', {'username': 'gdsfds', 'password': 'sasa'})
#     ans = str(a) if a else 'zero'
#     print(ans)
#     await message.answer(
#         # await get_user_anime_list('jiga-jiga')
#         f'{round(time.time() - time_conrol, 2)} {ans} {message.text}'
#     )

# протестировать очередь на фастапи рабтоает не fifo, но пойдет, если заспамить минут 5 отвечать будет
# executor.start_polling(dp, skip_updates=True)


# logger = logging.getLogger(__name__)


async def main():
    # Initialize bot and dispatcher
    # Initialize bot and dispatcher
    API_TOKEN = os.environ.get("API_TOKEN")
    if not API_TOKEN:
        raise ValueError("Didn't find API_TOKEN in environment variables.")
    bot = Bot(token=API_TOKEN)

    storage = MongoStorage(host='localhost', port=27017, db_name='aiogram_fsm')
    dp = Dispatcher(bot, storage=storage)
    # dp = Dispatcher(bot, storage=MemoryStorage())

    dp.middleware.setup(LoggingMiddleware())
    # Настройка логирования в stdout
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    # )
    # logger.error("Starting bot")

    # Парсинг файла конфигурации
    # config = load_config("config/bot.ini")

    # Объявление и инициализация объектов бота и диспетчера
    # bot = Bot(token=API_TOKEN)
    # dp = Dispatcher(bot, storage=MemoryStorage())

    # Регистрация хэндлеров
    register_handlers_common(dp)
    register_handlers_message(dp)

    # Установка команд бота
    # await set_commands(bot)

    # Запуск поллинга
    # await dp.skip_updates()  # пропуск накопившихся апдейтов (необязательно)
    await dp.start_polling()

    await dp.storage.close()
    await dp.storage.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())