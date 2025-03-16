from .check_arrays import check_array_and_process_nan_2d, \
                          flatten_multi_dimension_array_2d
from meteotools.exceptions import DimensionError
from numpy import ndarray
import numpy as np
import ctypes as ct
from meteotools import path_of_meteotools, platform_name


cint = ct.c_int
cdouble = ct.c_double
cdouble_p = ct.POINTER(ct.c_double)
if platform_name == 'Windows':
    fastcompute = ct.CDLL(path_of_meteotools + '/lib/fastcompute.dll')
elif platform_name == 'Linux':
    fastcompute = ct.CDLL(path_of_meteotools + '/lib/fastcompute.so')
fastcompute.interp2D_fast.argtypes = [cint, cint,
                                    cdouble, cdouble,
                                    cint, cdouble_p,
                                    cdouble_p,
                                    cdouble_p,
                                    cdouble_p]
fastcompute.interp2D_fast.restype = None
fastcompute.interp2D_fast_layers.argtypes = [cint, cint, cint,
                                    cdouble, cdouble,
                                    cint, cdouble_p,
                                    cdouble_p,
                                    cdouble_p,
                                    cdouble_p]
fastcompute.interp2D_fast_layers.restype = None


@check_array_and_process_nan_2d
def interp2D_fast(dx: float, dy: float,
                  xLoc: ndarray, yLoc: ndarray,
                  data: ndarray) -> ndarray:
    '''
    將等間距的2D直角坐標雙線性內插至任意位置
    data的各方向網格間距必須相等

    Parameters
    ----------
    dx : data 第1維(X)的直角網格資料間距
    dy : data 第0維(Y)的直角網格資料間距
    xLoc : 要內插的目標點第1維(X)位置
    yLoc : 要內插的目標點第0維(Y)位置
    data : 2維的各方向等間距的網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後新位置的陣列，維度與xLoc、yLoc相同

    '''
    if len(data.shape) == 2:
        original_loc_shape = xLoc.shape
        xLoc = xLoc.reshape(-1)
        yLoc = yLoc.reshape(-1)
        lenx = data.shape[1]
        leny = data.shape[0]
        N = xLoc.shape[0]

        original_data_dtype = data.dtype
        if original_data_dtype != np.float64:
            data = data.astype(np.float64)

        if data.flags.f_contiguous != True:
            data = np.asfortranarray(data)

        output = np.empty(N)

        fastcompute.interp2D_fast(ct.c_int(lenx),ct.c_int(leny),ct.c_double(dx),
                            ct.c_double(dy),ct.c_int(N),
                            xLoc.ctypes.data_as(cdouble_p),
                            yLoc.ctypes.data_as(cdouble_p),
                            data.ctypes.data_as(cdouble_p),
                            output.ctypes.data_as(cdouble_p))


        return output.reshape(original_loc_shape)

    else:
        raise DimensionError("來源資料(data)維度需為2")


@check_array_and_process_nan_2d
@flatten_multi_dimension_array_2d
def interp2D_fast_layers(dx: float, dy: float,
                         xLoc: ndarray, yLoc: ndarray,
                         data: ndarray) -> ndarray:
    '''
    將等間距的2D直角坐標雙線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後兩個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(xLoc) = (g,h,i)
         dim(yLoc) = (g,h,i)
         dim(output) = (a,b,c,d,g,h,i)
    (e,f) 為內插的維度，內插至(g,h,i)的位置
    (a,b,c,d)如時間、高度，沿這些維度重複內插(e,f)資料至(g,h,i)位置

    Parameters
    ----------
    dx : data 第1維(X)的直角網格資料間距
    dy : data 第0維(Y)的直角網格資料間距
    xLoc : 要內插的目標點第1維(X)位置
    yLoc : 要內插的目標點第0維(Y)位置
    data : 2維的等間距直角網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後的陣列
    '''

    # 維度處理，包含輸出的output維度、內插的layer層數
    original_loc_shape = xLoc.shape
    xLoc = xLoc.reshape(-1)
    yLoc = yLoc.reshape(-1)

    lenx = data.shape[2]
    leny = data.shape[1]
    N = xLoc.shape[0]
    layers = data.shape[0]

    original_data_dtype = data.dtype
    if original_data_dtype != np.float64:
        data = data.astype(np.float64)

    if data.flags.f_contiguous != True:
        data = np.asfortranarray(data)

    output = np.empty([layers,N])
    output = np.asfortranarray(output)

    fastcompute.interp2D_fast_layers(ct.c_int(lenx),ct.c_int(leny),
                                     ct.c_int(layers),ct.c_double(dx),
                                    ct.c_double(dy),ct.c_int(N),
                                    xLoc.ctypes.data_as(cdouble_p),
                                    yLoc.ctypes.data_as(cdouble_p),
                                    data.ctypes.data_as(cdouble_p),
                                    output.ctypes.data_as(cdouble_p))


    return output.reshape([data.shape[0]] + list(original_loc_shape))
