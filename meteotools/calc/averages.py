import numpy as np
from .check_arrays import pre_process_for_2Daverage, pre_process_for_1Daverage, change_axis


def convert_nan_to_default_value(axis, min_value, max_value):
    if np.isnan(min_value):
        min_value = axis[0]
    if np.isnan(max_value):
        max_value = axis[-1]
    return min_value, max_value


@pre_process_for_2Daverage
def calc_RZaverage(var: np.ndarray, z_axis: np.ndarray, r_axis: np.ndarray,
                   rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan,
                   r_weight=True) -> float:
    '''
    計算RZ座標下(垂直向對徑向，如颱風軸對稱系統)的特定徑向範圍、垂直向範圍平均值

    Parameters
    ----------
    var : 要被計算平均的變數
    z_axis : 垂直向座標，座標必須為嚴格遞增
    r_axis : 徑向座標，座標必須為嚴格遞增
    rmin : 徑向範圍起點，若未給定，則為徑向座標第一個點r_axis[0]
    rmax : 徑向範圍終點，若未給定，則為徑向座標最後一個點r_axis[-1]
    zmin : 徑向範圍起點，若未給定，則為徑向座標第一個點z_axis[0]
    zmax : 徑向範圍終點，若未給定，則為徑向座標最後一個點z_axis[-1]
    r_weight : 是否進行徑向加權平均，預設為True

    Raises
    ------
    ValueError : z_axis或r_axis不是嚴格遞增

    Returns
    -------
    取完RZ範圍平均之數值 float
    '''

    rmin, rmax = convert_nan_to_default_value(r_axis, rmin, rmax)
    zmin, zmax = convert_nan_to_default_value(z_axis, zmin, zmax)

    z_select = np.logical_and(z_axis >= zmin, z_axis <= zmax)
    r_select = np.logical_and(r_axis >= rmin, r_axis <= rmax)

    tmp = np.nanmean(var[z_select], axis=0)  # 垂直向
    # 徑向
    if r_weight == True:
        varout = np.nansum(tmp*r_select*r_axis)/np.nansum(r_select*r_axis)
    else:
        varout = np.nanmean(tmp[r_select])
    return varout


@pre_process_for_1Daverage
def calc_Raverage(var: np.ndarray, r_axis: np.ndarray, axis: int,
                  rmin=np.nan, rmax=np.nan) -> np.ndarray:
    '''
    計算圓柱座標下的特定徑向範圍平均值(有考慮徑向加權)

    Parameters
    ----------
    var : 要被計算平均的變數，可為多維
    r_axis : 徑向座標，座標必須為嚴格遞增
    axis : 要取徑向加權平均的維度
    rmin : 徑向範圍起點，若未給定，則為徑向座標第一個點r_axis[0]
    rmax : 徑向範圍終點，若未給定，則為徑向座標最後一個點r_axis[-1]

    Raises
    ------
    ValueError : r_axis不是嚴格遞增

    Returns
    -------
    取完徑向加權平均的陣列
    '''

    rmin, rmax = convert_nan_to_default_value(r_axis, rmin, rmax)

    r_select = np.logical_and(r_axis >= rmin, r_axis <= rmax)

    varout = np.nansum(var*r_select*r_axis, axis=-1) \
        / np.nansum(r_select*r_axis)
    return varout


@pre_process_for_1Daverage
def calc_Zaverage(var: np.ndarray, z_axis: np.ndarray, axis: int,
                  zmin=np.nan, zmax=np.nan) -> np.ndarray:
    '''
    計算圓柱座標下的特定垂直向範圍平均值

    Parameters
    ----------
    var : 要被計算平均的變數，可為多維
    z_axis : 垂直向座標，座標必須為嚴格遞增
    axis : 要取垂直向平均的維度
    zmin : 垂直向範圍起點，若未給定，則為垂直向座標第一個點r_axis[0]
    zmax : 垂直向範圍終點，若未給定，則為垂直向座標最後一個點r_axis[-1]

    Raises
    ------
    ValueError : r_axis不是嚴格遞增

    Returns
    -------
    取完垂直向平均的陣列
    '''
    zmin, zmax = convert_nan_to_default_value(z_axis, zmin, zmax)

    z_select = np.logical_and(z_axis >= zmin, z_axis <= zmax)

    varout = np.nansum(var*z_select, axis=-1)/np.sum(z_select)
    return varout
