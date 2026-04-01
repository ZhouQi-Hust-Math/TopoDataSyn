__all__ = [
    'DIY_dataset',
    'Function',
    'NN_model',
    'Plot_command',
]


from importlib import import_module
import re
from datetime import datetime


# Please keep this list sorted
assert __all__ == sorted(__all__)


def is_valid_timestr(time_str: str) -> bool:
    pattern = r"^\d{6}-\d{2}:\d{2}$"
    if not re.match(pattern, time_str):
        # "格式错误，应为 yymmdd-HH:MM"
        raise ValueError(f"格式错误，应为 yymmdd-HH:MM: {time_str}")

    try:
        # "格式正确"
        datetime.strptime(time_str, "%y%m%d-%H:%M")
        return True

    except ValueError:
        # "不是合法的日期时间"
        raise ValueError(f"不是合法的日期时间: {time_str}")


class version_register:
    def __init__(self, timeversion:str):
        assert is_valid_timestr(timeversion)
        self.__time_version = timeversion

    def get_timeversion(self):
        return self.__time_version


def get_all_timeversions():
    results = {}
    for module_name in __all__:
        try:
            module = import_module(f"{__name__}.{module_name}")

            version_class = getattr(module, "version_info", None)
            if version_class is None:
                print(f"{module_name} 中没有 version_info")
                continue

            version_instance = version_class()
            get_timeversion = getattr(version_instance, "get_timeversion", None)
            if get_timeversion is None:
                print(f"{module_name}.version_info 中没有 get_timeversion")
                continue

            results[module_name] = get_timeversion()

        except Exception as e:
            print(f"{module_name} 处理失败: {e}")

    return results


def check_timeversion(input_time, time_dict:dict):
    # 如果没有历史版本，默认认为它最大
    is_valid_timestr(input_time)
    if not time_dict:
        return True, None, None

    max_module = max(time_dict, key=time_dict.get)
    max_time = time_dict[max_module]
    return input_time > max_time, max_module, max_time


