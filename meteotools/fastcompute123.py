import numpy as np
import time as tm
from numpy import ndarray

if __name__ == 'meteotools.fastcompute':
    from .exceptions import InputError, DimensionError, LengthError
    from . import fastcompute as fc
else:
    from exceptions import InputError, DimensionError, LengthError
    import fastcompute as fc


def Make_cyclinder_coord(
        centerLocation: list, r: ndarray, theta: ndarray) -> ndarray:
    '''
    建立圓柱座標的水平格點在直角坐標上的位置

    Parameters
    ----------
    centerLocation : 圓柱座標中心在直角坐標上的 [y位置,x位置]
    r : 徑方向座標點
    theta : 切方向座標點 (rad，數學角)

    Returns
    -------
    圓柱座標的水平格點在直角座標上的位置 (theta,r)

    '''

    # 建立要取樣的座標位置(水平交錯)
    rr, ttheta = np.meshgrid(r, theta)
    xThetaRLocation = rr*np.cos(ttheta) + centerLocation[1]
    yThetaRLocation = rr*np.sin(ttheta) + centerLocation[0]

    return xThetaRLocation, yThetaRLocation


def cartesian2cylindrical(
        dx: float, dy: float, car_data: ndarray,
        centerLocation=None, r=None, theta=None,
        xTR=None, yTR=None) -> ndarray:
    '''
    將直角坐標內插到圓柱座標，只進行水平上的內插，並延續直角座標的垂直方向
    直角座標水平方向網格間距必須相等。

    Parameters
    ----------
    dx : data 第2維(X)的網格間距
    dy : data 第1維(Y)的網格間距
    cartesian_data : 直角座標資料

    Option
    ------
    要建立的圓柱座標的座標資訊，請輸入centerLocation, r, theta 或 xTR, yTR
    centerLocation : 圓柱座標中心在直角坐標上的 [y位置,x位置]
    r : 徑方向座標點 (ndarray)
    theta : 切方向座標點 (rad，數學角) (ndarray)
    xTR : 圓柱座標水平X位置 (theta,r) (ndarray)
    yTR : 圓柱座標水平Y位置 (theta,r) (ndarray)

    Raises
    ------
    InputError : 不足的座標資訊
    DimensionError :  資料維度錯誤

    Returns
    -------
    cyl_data : 內插後的圓柱座標資料 (z, theta, r)

    '''
    if len(cartesian_data.shape) != 3:
        raise DimensionError('直角座標資料(car_data)須為3維')

    def check_dimension(variable):

    def check_input(centerLocation, r, theta, xTR, yTR):
        if xTR != None and yTR != None:
            return xTR, yTR
        elif centerLocation != None and r != None and theta != None:
            xTR, yTR = Make_cyclinder_coord(centerLocation, r, theta)
            return xTR, yTR
        else:
            raise InputError("(xTR,yTR) 或 (centerLocation,r,theta) 須至少輸入一者")

    xTR, yTR = check_input(centerLocation, r, theta, xTR, yTR)

    cyl_data = interp2D_fast_layers(dx, dy, xTR, yTR, cartesian_data)
    return cyl_data
