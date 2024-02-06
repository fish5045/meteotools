import numpy as np
from numpy import ndarray
from meteotools.interpolation import interp2D_fast_layers
from meteotools.exceptions import InputError, UnitError
from .check_arrays import check_dimension


def unit_ratio(in_unit):
    if in_unit == 'km':
        return 1000
    elif in_unit == 'hPa':
        return 100
    elif in_unit == 'deg':
        return np.pi/180
    else:
        return 1


def Make_vertical_axis(height_start: float, height_end: float,
                       dz: float, in_unit='m') -> np.ndarray:
    '''
    建立垂直座標，可指定單位 (輸出為SI單位)

    Parameters
    ----------
    height_start : 起使高度
    height_end : 結束高度
    dz : 網格間距
    in_unit : 單位，可為m, Pa, km, hPa

    Raises
    ------
    輸入了不支援的單位

    Returns
    -------
    垂直座標 (SI unit)
    '''
    # 檢查單位、單位轉換
    def check_unit():
        if in_unit not in ['m', 'Pa', 'km', 'hPa']:
            raise UnitError('不支援的單位')

    check_unit()
    ratio = unit_ratio(in_unit)
    height = np.arange(height_start, height_end+0.00001, dz, dtype='float64')

    return height*ratio


def Make_tangential_axis(theta_start: float, theta_end: float, dtheta: float,
                         in_unit='rad') -> np.ndarray:
    '''
    建立垂直座標，可指定單位 (輸出為SI單位)

    Parameters
    ----------
    theta_start : 起使切向角度
    theta_end : 結束切向角度
    dtheta : 網格間距
    in_unit : 單位，可為rad, deg

    Raises
    ------
    輸入了不支援的單位

    Returns
    -------
    切向座標 (SI unit)
    '''

    # 檢查單位、單位轉換
    def check_unit():
        if in_unit not in ['deg', 'rad']:
            raise UnitError('不支援的單位')

    check_unit()
    ratio = unit_ratio(in_unit)
    theta = np.arange(theta_start, theta_end+0.00001, dtheta, dtype='float64')
    if (theta_end - theta_start)*ratio == 2*np.pi:
        theta = theta[:-1]

    return theta*ratio


def Make_radial_axis(r_start: float, r_end: float, dr: float,
                     in_unit='m') -> np.ndarray:
    '''
    建立垂直座標，可指定單位 (輸出為SI單位)

    Parameters
    ----------
    r_start : 起使徑向位置
    r_end : 結束徑向位置
    dr : 網格間距
    in_unit : 單位，可為m, km

    Raises
    ------
    輸入了不支援的單位

    Returns
    -------
    徑向座標 (SI unit)
    '''

    # 檢查單位、單位轉換
    def check_unit():
        if in_unit not in ['m', 'km', ]:
            raise UnitError('不支援的單位')

    check_unit()
    ratio = unit_ratio(in_unit)
    # 建立徑向座標(圓柱座標)
    r = np.arange(r_start, r_end+0.0001, dr, dtype='float64')
    return r*ratio


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
        dx: float, dy: float, cartesian_data: ndarray,
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

    def check_input(xTR, yTR):
        if type(None) not in [type(xTR), type(yTR)]:
            return xTR, yTR
        elif type(None) not in [type(centerLocation), type(r), type(theta)]:
            xTR, yTR = Make_cyclinder_coord(centerLocation, r, theta)
            return xTR, yTR
        else:
            raise InputError("(xTR,yTR) 或 (centerLocation,r,theta) 須至少輸入一者")

    check_dimension(cartesian_data, 3)
    xTR, yTR = check_input(xTR, yTR)

    cyl_data = interp2D_fast_layers(dx, dy, xTR, yTR, cartesian_data)
    return cyl_data
