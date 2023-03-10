import numpy as np


def wswd_to_uv(ws, wd):
    '''
    給入風速風向，轉為東西、南北風，可為float或陣列。

    Parameters
    ----------
    ws : 風速
    wd : 風向 (deg)

    Returns
    -------
    u : 東西風
    v : 南北風

    '''

    u = -np.sin(np.deg2rad(wd)) * ws
    v = -np.cos(np.deg2rad(wd)) * ws
    return u, v


def uv_to_wswd(u, v):
    '''
    給入向東西、南北風，轉為風速風，可為float或陣列。

    Parameters
    ----------
    u : 東西風
    v : 南北風

    Returns
    -------
    ws : 風速
    wd : 風向 (deg)

    '''

    ws = np.sqrt((u * u) + (v * v))
    wd = (np.arctan2(-u, -v) * 180. / np.pi) % 360
    return ws, wd
