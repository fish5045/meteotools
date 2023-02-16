'''
檔名處理、檔案讀寫的模組包
'''


import time as tm
from pathlib import Path


def Get_wrfout_time(yr,mo,day,hr,mi,sc,shift):
    '''
    將wrf輸出設定檔(data_settings_file)所提供之時間訊息轉為wrfout檔名格式
    '''
    currSec = tm.mktime(tm.struct_time((yr,mo,day,hr,mi,sc,0,0,0)))
    return tm.strftime('%Y-%m-%d_%H_%M_%S',tm.localtime(currSec+shift))

def Get_time_sec(yr,mo,day,hr,mi,sc,shift):
    '''
    將wrf輸出設定檔(data_settings_file)所提供之時間訊息轉為wrfout檔名格式
    '''
    return tm.mktime(tm.struct_time((yr,mo,day,hr,mi,sc,0,0,0)))


def mkdir(dirpath):
    '''
    建立資料夾，並忽略資料夾存在的錯誤
    可給一連串路徑建立一串資料夾，如 "test/a/b/c"
    '''
    Path(dirpath).mkdir(parents=True, exist_ok=True)


