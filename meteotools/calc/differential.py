import numpy as np
from .check_arrays import pre_post_process_for_differential
from scipy.fft import fft, ifft, fftfreq


@pre_post_process_for_differential(0)
def FD2(var: np.ndarray, delta: float, axis: int, cyclic=False):
    '''
    以二階中差分法（邊界使用二階偏差分法）計算變數var的微分值。

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取微分的維度。
    cyclic : bool, optional
        此方向是否是循環而週期性的（如圓柱座標的切方向）
        預設為False。

    Returns
    -------
    output : 二階差分法微分值，陣列大小與var相同。
    '''

    # 此方向為週期性
    if cyclic == True:
        output = (np.roll(var, -1, 0)-np.roll(var, 1, 0))/2/delta

    # 非週期性
    elif cyclic == False:
        output = np.zeros(var.shape)*np.nan
        output[0] = (-3*var[0]+4*var[1]-var[2])/2/delta
        output[1:-1] = (var[2:]-var[:-2])/2/delta
        output[-1] = (3*var[-1]-4*var[-2]+var[-3])/2/delta

    return output

@pre_post_process_for_differential(0)
def FD4(var: np.ndarray, delta: float, axis: int, cyclic=False):
    '''
    以四階中差分法計算變數var的微分值。
    var[2:-2] 以4階中差分法計算
    var[1] var[-2] 以2階中差分法計算
    var[0] var[-1] 以2階偏差分法計算

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取微分的維度。
    cyclic : bool, optional
        此方向是否是循環而週期性的（如圓柱座標的切方向）
        預設為False。

    Returns
    -------
    output : 四階差分法微分值，陣列大小與var相同。
    '''

    # 此方向為週期性
    if cyclic == True:
        output = (-np.roll(var, -2, 0) + 8*np.roll(var, -1, 0)
                  - 8*np.roll(var, 1, 0) + np.roll(var, 2, 0))/12/delta

    # 非週期性
    elif cyclic == False:
        output = np.zeros(var.shape)*np.nan
        output[0] = (-3*var[0]+4*var[1]-var[2])/2/delta
        output[1] = (var[2]-var[0])/2/delta
        output[2:-2] = (-var[4:] + 8*var[3:-1] - 8*var[1:-3] + var[:-4])/12/delta
        output[-2] = (var[-1]-var[-3])/2/delta
        output[-1] = (3*var[-1]-4*var[-2]+var[-3])/2/delta

    return output

@pre_post_process_for_differential(0)
def FD2_front(var: np.ndarray, delta: float, axis: int,) -> np.ndarray:
    '''
    以二階前差分法計算變數var的微分值（下邊界兩排為nan）。

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取微分的維度。

    Returns
    -------
    output : 二階前差分法微分值，陣列大小與var相同。
    '''

    output = np.zeros(var.shape)*np.nan
    output[:-2] = (-3*var[:-2]+4*var[1:-1]-var[2:])/2/delta
    return output


@pre_post_process_for_differential(0)
def FD2_back(var: np.ndarray, delta: float, axis: int,) -> np.ndarray:
    '''
    以二階後差分法計算變數var的微分值（上邊界兩排為nan）。

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取微分的維度。

    Returns
    -------
    output : 二階後差分法微分值，陣列大小與var相同。
    '''

    output = np.zeros(var.shape)*np.nan
    output[2:] = (3*var[2:]-4*var[1:-1]+var[:-2])/2/delta
    return output


@pre_post_process_for_differential(0)
def FD2_2(var: np.ndarray, delta: float, axis: int,) -> np.ndarray:
    '''
    以二階中差分法計算變數var的二次微分值（邊界點為偏差分法，準確度二階）。

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取二次微分的維度。

    Returns
    -------
    output : 二階中差分法二次微分值，陣列大小與var相同。。
    '''

    output = np.zeros(var.shape)*np.nan
    output[0] = (2*var[0]-5*var[1]+4*var[2]-var[3])/delta/delta
    output[1:-1] = (var[2:]-2*var[1:-1]+var[:-2])/delta/delta
    output[-1] = (2*var[-1]-5*var[-2]+4*var[-3]-var[-4])/delta/delta
    return output


@pre_post_process_for_differential(0)
def FD2_2_front(var: np.ndarray, delta: float, axis: int,) -> np.ndarray:
    '''
    以二階前差分法計算變數var的二次微分值（下邊界兩排為nan）。

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取二次微分的維度。

    Returns
    -------
    output : 二階前差分法二次微分值，陣列大小與var相同。
    '''

    output = np.zeros(var.shape)*np.nan
    output[:-3] = (2*var[:-3]-5*var[1:-2]+4*var[2:-1]-var[3:])/delta/delta
    return output


@pre_post_process_for_differential(0)
def FD2_2_back(var: np.ndarray, delta: float, axis: int,) -> np.ndarray:
    '''
    以二階後差分法計算變數var的二次微分值（上邊界兩排為nan）。

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取二次微分的維度。

    Returns
    -------
    output : 二階後差分法二次微分值，陣列大小與var相同。。
    '''

    output = np.zeros(var.shape)*np.nan
    output[3:] = (2*var[3:]-5*var[2:-1]+4*var[1:-2]-var[:-3])/delta/delta
    return output


@pre_post_process_for_differential(-1)
def difference_FFT(var, delta, axis):
    '''
    以傅立葉變換FFT計算變數var的微分值。

    Parameters
    ----------
    var : 要計算差分的陣列。
    delta : 此維度的網格間距。
    axis : 要取微分的維度。

    Returns
    -------
    output : 傅立葉變換找出的微分值，陣列大小與var相同。
    '''

    yf = fft(var)*2*np.pi/delta
    n = yf.shape[-1]
    yf *= 1j*fftfreq(n)
    output = ifft(yf)
    return output
