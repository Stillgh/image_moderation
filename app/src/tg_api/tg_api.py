import os

from telebot import types, TeleBot

from database.database import get_session
from service.crud.user_service import get_all_users


class TgBot:
    def __init__(self):
        self.bot = None

    def setup(self):
        self.bot = TeleBot(os.getenv('TG_TOKEN'))
        self._setup_handlers()

    def start_polling(self):
        self.bot.infinity_polling()

    def _setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def start_bot(message):
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            users_btn = types.KeyboardButton('👥 List Users')
            markup.add(users_btn)

            welcome_text = (
                f"Hello, {message.from_user.first_name}!\n"
                "Click the button to see registered users."
            )
            self.bot.reply_to(message, welcome_text, reply_markup=markup)

        @self.bot.message_handler(func=lambda message: message.text == '👥 List Users')
        def list_users(message):
            try:
                with next(get_session()) as session:
                    users = get_all_users(session)
                    if not users:
                        self.bot.reply_to(message, "No registered users found.")
                        return

                    user_list = "📋 Registered Users:\n\n"
                    for user in users:
                        user_list += (
                            f"👤 {user.name} {user.surname}\n"
                            f"📧 {user.email}\n"
                            f"💰 Balance: {user.balance}\n"
                            f"-------------------\n"
                        )

                    self.bot.reply_to(message, user_list)
                    print("User list requested")

            except Exception as e:
                error_msg = f"Error fetching users: {str(e)}"
                print(error_msg)
                self.bot.reply_to(message, "❌ Error fetching user list")

