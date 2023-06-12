from aiogram.dispatcher import Dispatcher
from .handlers import *

def register_handlers_common(dp: Dispatcher):
    dp.register_message_handler(cmd_start, commands="start", state="*")
    dp.register_message_handler(cmd_test, commands="test", state="*")
    dp.register_message_handler(cmd_recommend_again, commands="recommend_again", state="*")
    dp.register_message_handler(cmd_change_username, commands='change_username', state="*")
    dp.register_message_handler(cmd_feedback, commands='feedback', state="*")
    dp.register_message_handler(cmd_help, commands='help', state="*")
    dp.register_message_handler(cmd_donate, commands='donate', state="*")
    dp.register_message_handler(cmd_about, commands='about', state="*")

def register_handlers_message(dp: Dispatcher):
    dp.register_message_handler(username_got, state=Pipeline.waiting_username)
    dp.register_message_handler(ending, state=Pipeline.waiting_feedback)
    dp.register_callback_query_handler(confirm_username, text='confirm_username',
                                       state=Pipeline.waiting_username_confirmation)
    dp.register_callback_query_handler(cancel_username, text='cancel_username',
                                       state=Pipeline.waiting_username_confirmation)
    dp.register_callback_query_handler(choice_anime,
                                       state=Pipeline.waiting_explanation_choice)
    dp.register_message_handler(unbear, state=[Pipeline.executing_checking_username,
                                               Pipeline.executing_recommender_pipeline,
                                               ])