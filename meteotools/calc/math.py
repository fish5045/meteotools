
import numpy as np
from .check_arrays import change_axis

def find_root2(a, b, c):
    '''
    求解 ax^2 + bx + c = 0 solve x 的根，可為float或陣列。

    Parameters
    ----------
    a : 二次項係數
    b : 一次項係數
    c : 常數項

    Returns
    -------
    x1 : 正根(complex if b^2-4ac < 0)
    x2 : 負根(complex if b^2-4ac < 0)
    '''
    x1 = (-b+(b**2-4*a*c)**0.5)/(2*a)
    x2 = (-b-(b**2-4*a*c)**0.5)/(2*a)
    return x1, x2



def nine_points_smooth(var: np.ndarray, center_weight=1.):
    '''
    九點平滑法，針對各網格點進行以下計算(center_weight以cw簡寫):
             1     ╔ 1  1  1 ╗╔ .   .   . ╗
    out = ──────── ║ 1 cw  1 ║║ .  var  . ║
           8 + cw  ╚ 1  1  1 ╝╚ .   .   . ╝

    Parameters
    ----------
    var : 要被平滑的陣列
    center_weight : 中央格點的權重

    Returns
    -------
    out : 九點平滑法後的陣列
    '''
    svar = np.zeros([var.shape[0]+2, var.shape[1]+2])
    svar[1:-1, 1:-1] = var
    svar[0, 1:-1] = var[0, :]
    svar[-1, 1:-1] = var[-1, :]
    svar[1:-1, 0] = var[:, 0]
    svar[1:-1, -1] = var[:, -1]
    svar[0, 0] = var[0, 0]
    svar[-1, 0] = var[-1, 0]
    svar[0, -1] = var[0, -1]
    svar[-1, -1] = var[-1, -1]

    out = (svar[:-2, :-2] + svar[:-2, 1:-1] + svar[:-2, 2:]
           + svar[1:-1, :-2] + svar[1:-1, 1:-1]*center_weight + svar[1:-1, 2:]
           + svar[2:, :-2] + svar[2:, 1:-1] + svar[2:, 2:])/(8+center_weight)
    return out

@change_axis(0)
def three_points_smooth(var, axis, center_weight=1.):
    '''
    三點平滑法，針對各網格點進行以下計算(center_weight以cw簡寫):
             1                ╔  .  ╗
    out = ──────── [ 1 cw  1 ]║ var ║
           2 + cw             ╚  .  ╝

    Parameters
    ----------
    var : 要被平滑的陣列
    axis : 要平滑處理的維度
    center_weight : 中央格點的權重

    Returns
    -------
    out : 九點平滑法後的陣列
    '''

    svar = np.zeros([var.shape[0], var.shape[1]+2])
    svar[1:-1] = var
    svar[0] = var[0]
    svar[-1] = var[-1]
    out = (svar[:-2] + svar[1:-1] + svar[2:])/3.
    return out


