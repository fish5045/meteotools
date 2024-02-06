import os
import numpy as np
from os.path import join as pth
import cv2


def movie_make(src_dir, video_name, fps=30, video_fmt='mp4v'):
    '''
    將一系列的圖片轉為mp4影片，圖片路徑不能包含中文
    輸入: 圖片的資料夾       src_dir       str
         影片路徑與檔名      video_name    str
    選項: 畫面更新率         fps           30
         影片格式           video_fmt     'aviv'

    '''
    c = os.listdir(src_dir)
    c.sort(key=lambda e: int(e.split('.')[0]))
    pic = [_pic for _pic in c if '.png' in _pic]

    try:
        size = cv2.imread(pth(src_dir, pic[0])).shape
    except:
        print('fail')
        return

    videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
        *video_fmt), fps, (size[1], size[0]))
    for indx, _pic in enumerate(pic):
        imge = cv2.imread(pth(src_dir, _pic), cv2.IMREAD_COLOR)
        videoWriter.write(imge)
        print(f'\r{indx} : {_pic}', end='')
    videoWriter.release()
    print('\nCompleted making video')
