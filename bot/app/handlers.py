import asyncio
import json
import logging
from io import BytesIO

from aiogram import Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from .my_utils import *
from aiogram.types.chat import ChatActions
from pathlib import Path
import aiofiles
import base64

async def __init__():
    pass


audios = data_path.joinpath('audio')
voices = {}

class Pipeline(StatesGroup):
    pre_start = State()
    waiting_username = State()
    executing_checking_username = State()
    about = State()
    change_username = State()
    choice_another_username = State()
    waiting_username_confirmation = State()
    executing_recommender_pipeline = State()
    setup_profile = State()
    waiting_explanation_choice = State()
    pre_feedback = State()
    waiting_feedback = State()
    ending = State()
    null = State()


@add_event
async def cmd_about(message: types.Message, state: FSMContext):
    url = 'https://content.artofmanliness.com/uploads/2022/02/no-header.jpg'

    buttons = [
        types.InlineKeyboardButton(text="Repository with good description", url=url),
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)

    await send_text(message, text='Here you can read about the recommendation algorithm and the bot architecture:',
                    reply_markup=keyboard)


@add_event
async def cmd_donate(message: types.Message, state: FSMContext):
    url = 'https://www.patreon.com/iskander687/membership'

    buttons = [
        types.InlineKeyboardButton(text="Patreon", url=url),
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)

    await send_text(message, text='Creator would be grateful if you support this project)', reply_markup=keyboard)


@add_event
async def cmd_help(message: types.Message, state: FSMContext):
    await send_text(message,
                    text=("/donate - support the author\n"
                          "/start - start again\n"
                          "/help - get this message\n"
                          "/feedback - give feedback\n"
                          "/about - how recommendations work\n"
                          "/recommend_again - get recs for the same username\n"
                          "/change_username - guess wat"))

@add_event
async def cmd_feedback(message: types.Message, state: FSMContext):
    await pre_feedback(message, state)


@add_event
async def cmd_change_username(message: types.Message, state: FSMContext):
    await state.set_state(Pipeline.waiting_username.state)
    await send_text(message, text='Send me your username on MyAnimeList:')

@add_event
async def cmd_recommend_again(message: types.Message, state: FSMContext):
    await recommender_pipeline(message, state)

@add_event
async def cmd_test(message: types.Message, state: FSMContext):
    await state.update_data(username='jiga-jiga')
    await state.set_state(Pipeline.executing_recommender_pipeline.state)
    await asyncio.sleep(6)
    await send_text(message, text='ass')


@add_event
async def cmd_start(message: types.Message, state: FSMContext):
    await state.set_state(Pipeline.pre_start.state)
    await set_typing_status(message,
                            pre_delay=1.1,
                            typing_time=6,
                            action_type=ChatActions.RECORD_VOICE)
    await set_typing_status(message,
                            pre_delay=0,
                            typing_time=1,
                            action_type=ChatActions.UPLOAD_VOICE)
    await send_voice(message, 'pre_start_1')

    await set_typing_status(message,
                            pre_delay=0.2,
                            typing_time=1,
                            action_type=ChatActions.CHOOSE_STICKER)
    await send_sticker(message, sticker="CAACAgIAAxkBAAEI2c5kU2OVcSoxqpCNEAgGj2-OHHUFrgACcBUAAjvjSEixUwLGviuHKy8E")
    # await asyncio.sleep(0.3)
    await state.set_state(Pipeline.waiting_username.state)
    await set_typing_status(message,
                            pre_delay=0.1,
                            typing_time=1,
                            action_type=ChatActions.TYPING)
    await send_text(message, text="Send me your username on [MyAnimeList](https://myanimelist.net/):",
                    parse_mode='Markdown',
                    disable_web_page_preview=True)
#     await message.answer('''
# Reo-senpai can suggest you anime recommendations.
# But first send me your username on MyAnimeList.net
# Kudasai:''')

@add_event
async def username_got(message: types.Message, state: FSMContext):
    await state.set_state(Pipeline.executing_checking_username.state)
    username = message.text.strip()
    await state.update_data(username=username)

    await set_typing_status(message,
                            pre_delay=0.2,
                            typing_time=2,
                            action_type=ChatActions.TYPING)
    await send_text(message, text="Arigato!\nWait a few seconds, I'll check it.")

    user_info = await get_user_info(username)

    if user_info == -1:
        await set_typing_status(message,
                                pre_delay=0,
                                typing_time=2,
                                action_type=ChatActions.TYPING)
        await send_text(message, text="MAL servers do not response(\nTry later by /change_username")
        await set_typing_status(message,
                                pre_delay=0.5,
                                typing_time=0.1,
                                action_type=ChatActions.CHOOSE_STICKER)
        await send_sticker(message, sticker="CAACAgIAAxkBAAEI-VBkYIDqokWCJGcmdLBg3vM3WqZcnAACTRYAAlG6QEitI8FkUYMMnC8E")
        return
    elif not user_info:
        await set_typing_status(message,
                                pre_delay=0,
                                typing_time=2,
                                action_type=ChatActions.TYPING)

        await state.set_state(Pipeline.waiting_username.state)
        await send_text(message, text=(f'I couldn\'t find {username} in MAL.\n'
                                       f'Please check if the username is correct or if the MAL profile is public.\n\n'
                                       f'Could you send me your username on MAL again, please:\n'))
        return

    keyboard = types.InlineKeyboardMarkup(row_width=2)
    buttons = []
    for text, data in [('Yes', 'confirm_username'), ('No', 'cancel_username')]:
        buttons.append(types.InlineKeyboardButton(text=text, callback_data=data))
    keyboard.add(*buttons)
    if user_info.get('image', None):
        await set_typing_status(message,
                                pre_delay=0,
                                typing_time=3,
                                action_type=ChatActions.UPLOAD_PHOTO)
        await state.set_state(Pipeline.waiting_username_confirmation.state)
        await send_photo(message, photo=user_info.get('image'),
            caption=f"That's what I found about {username}\n\n" +
                    '\n'.join([f'{i}: {j}' for i, j in user_info.items() if i not in ['image', 'username']]) +
                    '\n\nIs it your profile?', reply_markup=keyboard)
    else:
        await set_typing_status(message,
                                pre_delay=0,
                                typing_time=2,
                                action_type=ChatActions.TYPING)
        await state.set_state(Pipeline.waiting_username_confirmation.state)
        await send_text(message, text=f"That's what I found about {username}\n\n" +
                             '\n'.join([f'{i}: {j}' for i, j in user_info.items() if i not in ['image', 'username']]) +
                             '\n\nIs it your profile?', reply_markup=keyboard)

@add_event
async def confirm_username(call: types.CallbackQuery, state: FSMContext):
    await call.answer()
    await recommender_pipeline(call.message, state=state)



async def recommender_pipeline(message: types.Message, state: FSMContext):
    await state.set_state(Pipeline.executing_recommender_pipeline.state)
    await set_typing_status(message,
                            pre_delay=0.1,
                            typing_time=12,
                            action_type=ChatActions.RECORD_VOICE)
    await set_typing_status(message,
                            pre_delay=0,
                            typing_time=0.3,
                            action_type=ChatActions.UPLOAD_VOICE)
    await send_voice(message, 'executing_recommender_pipeline_1')
    await asyncio.sleep(17)
    username = (await state.get_data()).get('username', None)

    if not username:
        # logging.error('username error')
        await state.set_state(Pipeline.waiting_username.state)
        await send_text(message, text="Send me your username on MyAnimeList:")
        return

    recs = await call_model('http://127.0.0.1:1111/', {'username': username})

    if not recs:
        pass
    elif recs == -1:
        await send_text(message, text="Something went wrong, text to author. And try later by /recommend_again ...")
        await send_sticker(message, sticker="CAACAgIAAxkBAAEI-VBkYIDqokWCJGcmdLBg3vM3WqZcnAACTRYAAlG6QEitI8FkUYMMnC8E")
        return

    if 'error' in recs:
        # logging.error(f'Model ERROR: {recs["error"]}')
        if recs['error'] == 'no matching titles':
            await set_typing_status(message,
                                    pre_delay=0,
                                    typing_time=1,
                                    action_type=ChatActions.TYPING)
            await send_text(message, text="Reo-senpai can't suggest anything for you. Because she couldn't find anime in your animelist that she knows. "
                                          "Don't be upset. U can try update your animelist and try again by /recommend_again , \nalso U can try get recs for another profiles by /change_username")
            await set_typing_status(message,
                                    pre_delay=0.5,
                                    typing_time=0.1,
                                    action_type=ChatActions.CHOOSE_STICKER)
            await send_sticker(message, sticker="CAACAgIAAxkBAAEI-UtkYH9Wx-zk3D2hHEpaK6zLfDASBwACOxYAAvXQth0d71ge99TKZy8E")

            return
        elif recs['error'] == 'mal server sucks':
            await set_typing_status(message,
                                    pre_delay=0,
                                    typing_time=2,
                                    action_type=ChatActions.TYPING)
            await send_text(message, text="MAL servers do not response(\nTry later by /recommend_again")
            await set_typing_status(message,
                                    pre_delay=0.5,
                                    typing_time=0.1,
                                    action_type=ChatActions.CHOOSE_STICKER)
            await send_sticker(sticker="CAACAgIAAxkBAAEI-VBkYIDqokWCJGcmdLBg3vM3WqZcnAACTRYAAlG6QEitI8FkUYMMnC8E")
            return
        await send_text(message, text="Sometthing went wrong, try later by /recommend_again ...")
        await send_sticker(message, sticker="CAACAgIAAxkBAAEI-VBkYIDqokWCJGcmdLBg3vM3WqZcnAACTRYAAlG6QEitI8FkUYMMnC8E")
        return


    await state.update_data(recs)
    await set_typing_status(message,
                            pre_delay=0,
                            typing_time=8,
                            action_type=ChatActions.RECORD_VOICE)
    await set_typing_status(message,
                            pre_delay=0,
                            typing_time=0.1,
                            action_type=ChatActions.UPLOAD_VOICE)
    await send_voice(message, 'executing_recommender_pipeline_2')
    await asyncio.sleep(10)
    await set_typing_status(message,
                            pre_delay=0.3,
                            typing_time=3,
                            action_type=ChatActions.UPLOAD_PHOTO)

    msg_str_0 = (f"**{recs['data'][0].get('title')}**\n"
               f"\n"
               f"Score: {recs['data'][0].get('score')}\n"
               f"Year: {recs['data'][0].get('year')}\n"
               f"Genres: {', '.join(recs['data'][0].get('genres'))}\n"
               f"Themes: {', '.join(recs['data'][0].get('themes'))}\n"
               f"[MAL page]({recs['data'][0].get('url')})")
    await send_photo(message, photo=recs['data'][0].get('image'),
                               caption=msg_str_0, parse_mode='Markdown')
    await asyncio.sleep(1)
    await send_text(message, text=f"What about this anime?\n{recs['data'][0].get('synopsis')}")
    await asyncio.sleep(1)
    await set_typing_status(message,
                            pre_delay=1,
                            typing_time=6,
                            action_type=ChatActions.UPLOAD_PHOTO)
    await send_photo(message, photo=BytesIO(base64.b64decode(recs.get('data')[0].get('explain_image'))))
    await set_typing_status(message,
                            pre_delay=0.1,
                            typing_time=3,
                            action_type=ChatActions.RECORD_VOICE)
    await set_typing_status(message,
                            pre_delay=0,
                            typing_time=0.2,
                            action_type=ChatActions.UPLOAD_VOICE)
    await send_voice(message, 'executing_recommender_pipeline_3', text="From top to bottom are the anime you've "
                                                                       "watched. The bars show the influence of the "
                                                                       "anime watched on the recommendation. The "
                                                                       "influence can be both positive and negative.")
    await asyncio.sleep(6)
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    for text, data in [(f'{i}. {val.get("title")}', i-1) for i, val in enumerate(recs['data'], start=1)]:
        keyboard.add(types.InlineKeyboardButton(text=text, callback_data=data))
    await state.set_state(Pipeline.waiting_explanation_choice.state)
    await set_typing_status(message,
                            pre_delay=11,
                            typing_time=2,
                            action_type=ChatActions.TYPING)
    await send_text(message, text="You can get an explanation of other suggestions, just click on it!", reply_markup=keyboard)


@add_event
async def cancel_username(call: types.CallbackQuery, state: FSMContext):
    await state.set_state(Pipeline.waiting_username.state)
    await send_text(call.message, text="Send me a correct username on MyAnimeList:")
    await call.answer()


@add_event
async def choice_anime(call: types.CallbackQuery, state: FSMContext):
    recs = await state.get_data()
    if call.data == 'ty':
        await pre_feedback(call.message, state)
        return
    i = int(call.data)

    msg_str = (f"**{recs['data'][i].get('title')}**\n"
               f"\n"
               f"Score: {recs['data'][i].get('score')}\n"
               f"Year: {recs['data'][i].get('year')}\n"
               f"Genres: {', '.join(recs['data'][i].get('genres'))}\n"
               f"Themes: {', '.join(recs['data'][i].get('themes'))}\n"
               f"[MAL page]({recs['data'][i].get('url')})")
    await send_photo(call.message, photo=recs['data'][i].get('image'),
                     caption=msg_str, parse_mode='Markdown')
    await asyncio.sleep(0.1)
    await send_text(call.message, text=f"Synopsis: {recs['data'][i].get('synopsis')}")
    await asyncio.sleep(0.1)
    await send_photo(call.message, photo=BytesIO(base64.b64decode(recs.get('data')[i].get('explain_image'))))
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    for text, data in [(f'{i}. {val.get("title")}', i - 1) for i, val in enumerate(recs['data'], start=1)]:
        keyboard.add(types.InlineKeyboardButton(text=text, callback_data=data))
    keyboard.add(types.InlineKeyboardButton(text='Thank you!', callback_data='ty'))
    await asyncio.sleep(0.1)
    await send_text(call.message, text="You can get an explanation of other suggestions, just click on it!",
                    reply_markup=keyboard)
    await call.answer()


async def pre_feedback(message: types.Message, state: FSMContext):
    await state.set_state(Pipeline.pre_feedback.state)
    await set_typing_status(message,
                            pre_delay=0.1,
                            typing_time=10,
                            action_type=ChatActions.RECORD_VOICE)
    await send_voice(message, 'pre_feedback_1')
    await set_typing_status(message,
                            pre_delay=0.5,
                            typing_time=1,
                            action_type=ChatActions.CHOOSE_STICKER)
    await send_sticker(message, sticker="CAACAgIAAxkBAAEI-WtkYJHkiX6LoNsdWIjXCoC16Ns6lAACtisAAulwQEmIpQ2j_xBfMy8E")
    await state.set_state(Pipeline.waiting_feedback.state)

@add_event
async def ending(message: types.Message, state: FSMContext):
    await state.set_state(Pipeline.ending.state)
    await state.update_data(feedback=message.text)
    await set_typing_status(message,
                            pre_delay=0.1,
                            typing_time=8,
                            action_type=ChatActions.RECORD_VOICE)
    await send_voice(message, 'ending_1')
    await asyncio.sleep(3)
    await state.set_state(Pipeline.null.state)
    await set_typing_status(message,
                            pre_delay=0.3,
                            typing_time=2,
                            action_type=ChatActions.TYPING)
    await send_text(message,
                    text=("/donate - support the author\n"
                          "/start - start again\n"
                          "/help - get this message\n"
                          "/feedback - give feedback\n"
                          "/about - how recommendations work\n"
                          "/recommend_again - get recs for the same username\n"
                          "/change_username - guess wat"))


@add_event
async def unbear(message: types.Message, state: FSMContext):
    async def sc1(message: Message):
        await set_typing_status(message,
                                pre_delay=0.1,
                                typing_time=1,
                                action_type=ChatActions.TYPING)
        await send_text(message, text="Please wait, your request is processing.")
        await set_typing_status(message,
                                pre_delay=0.5,
                                typing_time=0.1,
                                action_type=ChatActions.CHOOSE_STICKER)
        await send_sticker(message, sticker="CAACAgIAAxkBAAEI-YBkYJuJ_ayn5AHCRntn4EuPNjzJwwAC-SsAAtbSQUn2VetMXGhV2i8E")

    async def sc2(message: Message):
        await set_typing_status(message,
                                pre_delay=0.05,
                                typing_time=1,
                                action_type=ChatActions.RECORD_VOICE)
        await send_voice(message, 'unbear_1', text="What an impatient person! Please wait, we are processing your request.")


    async def sc3(message: Message):
        await set_typing_status(message,
                                pre_delay=0.05,
                                typing_time=1,
                                action_type=ChatActions.RECORD_VOICE)
        await send_voice(message, 'unbear_2')

    async def sc4(message: Message):
        await set_typing_status(message,
                                pre_delay=0.05,
                                typing_time=1,
                                action_type=ChatActions.RECORD_VOICE)
        await send_voice(message, 'unbear_3')

    scenarios = [sc1, sc2, sc3, sc4]

    if not (await state.get_data()).get('unbear', False):
        await state.update_data(unbear=0)
        i = 0
    else:
        i = (await state.get_data()).get('unbear')

    await scenarios[i](message)

    i = (i + 1) % len(scenarios)
    await state.update_data(unbear=i)