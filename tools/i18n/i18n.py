import json
import locale
import os


def load_language_list(language):
    with open(f"./i18n/locale/{language}.json", "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        # 默认语言设置为中文（zh_CN）
        default_language = "zh_CN"

        if language in ["Auto", None]:
            language = (
                default_language or locale.getdefaultlocale()[0]
            )  # 获取系统默认语言，如果无法识别则使用默认语言

        # 如果指定的语言文件不存在，则使用默认语言
        if not os.path.exists(f"./i18n/locale/{language}.json"):
            language = default_language

        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        # 根据键值返回对应的翻译，如果没有找到则返回键值本身
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
