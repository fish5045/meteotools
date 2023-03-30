'''
檔名處理、檔案讀寫的模組包
'''


from time import mktime, strftime, struct_time, localtime
from pathlib import Path
import json


def get_wrfout_time(year, month, day, hour, minute, second, offset_seconds):
    '''
    將wrf輸出設定檔(data_settings_file)所提供之時間訊息轉為wrfout檔名格式
    '''
    currSec = mktime(struct_time(
        (year, month, day, hour, minute, second, 0, 0, 0)))
    return strftime('%Y-%m-%d_%H_%M_%S', localtime(currSec+offset_seconds))


def get_time_sec(year, month, day, hour, minute, second, offset_seconds):
    '''
    將wrf輸出設定檔(data_settings_file)所提供之時間訊息轉為wrfout檔名格式
    '''
    seconds1970 = mktime(
        struct_time((year, month, day, hour, minute, second, 0, 0, 0)))
    return seconds1970 + offset_seconds


def mkdir(dirpath):
    '''
    建立資料夾，並忽略資料夾存在的錯誤
    可給一連串路徑建立一串資料夾，如 "test/a/b/c"
    '''
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def load_settings(file_path_and_name):
    with open(file_path_and_name, 'r') as f:
        settings = json.load(f)
    return settings
