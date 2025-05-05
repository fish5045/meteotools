from .check_arrays import check_array_and_process_nan_1d, \
    check_array_and_process_nan_1d_nonequal, \
    flatten_multi_dimension_array_1d
from meteotools.exceptions import DimensionError
from numpy import ndarray, zeros
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

fastcompute.interp1D_fast.argtypes = [cint,
                                    cdouble,
                                    cint, cdouble_p,
                                    cdouble_p,
                                    cdouble_p]
fastcompute.interp1D_fast.restype = None
fastcompute.interp1D_fast_layers.argtypes = [cint, cint,
                                    cdouble,
                                    cint, cdouble_p,
                                    cdouble_p,
                                    cdouble_p]
fastcompute.interp1D_fast_layers.restype = None
fastcompute.interp1D_nonequal.argtypes = [cint, cdouble_p,
                                    cint,
                                    cdouble_p, cdouble_p,
                                    cdouble_p,]
fastcompute.interp1D_nonequal.restype = None
fastcompute.interp1D_nonequal_layers.argtypes = [cint, cdouble_p,
                                    cint, cint,
                                    cdouble_p, cdouble_p,
                                    cdouble_p,]
fastcompute.interp1D_nonequal_layers.restype = None


@check_array_and_process_nan_1d
def interp1D_fast(dx: float, xLoc: ndarray, data: ndarray) -> ndarray:
    '''
    將等間距的1D直角坐標線性內插至任意位置
    data的網格間距必須相等

    Parameters
    ----------
    dx : data 第1維(X)的直角網格資料間距
    xLoc : 要內插的目標點第1維(X)位置
    data : 2維的等間距直角網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後新位置的陣列，維度與xLoc相同

    '''

    if len(data.shape) == 1:
        original_loc_shape = xLoc.shape
        xLoc = xLoc.reshape(-1)
        lenx = data.shape[0]
        N = xLoc.shape[0]

        original_data_dtype = data.dtype
        if original_data_dtype != np.float64:
            data = data.astype(np.float64)

        output = np.empty(N)

        fastcompute.interp1D_fast(ct.c_int(lenx),ct.c_double(dx),
                            ct.c_int(N),
                            xLoc.ctypes.data_as(cdouble_p),
                            data.ctypes.data_as(cdouble_p),
                            output.ctypes.data_as(cdouble_p))


        return output.reshape(original_loc_shape)
    else:
        raise DimensionError("來源資料(data)維度需為1")


@check_array_and_process_nan_1d_nonequal
def interp1D_nonequal(x_input: ndarray, x_output: ndarray, data: ndarray) -> ndarray:
    '''
    將不等間距的1D坐標線性內插至任意位置，data須為一維。

    Parameters
    ----------
    x_input : data 的原始格點位置
    x_output : 要內插的目標點位置
    data : 不等間距網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後的陣列
    '''

    if len(data.shape) == 1:
        original_loc_shape = x_output.shape
        x_output = x_output.reshape(-1)
        lenx = data.shape[0]
        N = x_output.shape[0]

        if data.dtype != np.float64:
            data = data.astype(np.float64)

        if x_input.dtype != np.float64:
            x_input = x_input.astype(np.float64)

        output = np.empty(N)

        fastcompute.interp1D_nonequal(ct.c_int(lenx),x_input.ctypes.data_as(cdouble_p),
                                  ct.c_int(N),x_output.ctypes.data_as(cdouble_p),
                                  data.ctypes.data_as(cdouble_p),
                                  output.ctypes.data_as(cdouble_p))


        return output.reshape(original_loc_shape)
    else:
        raise DimensionError("來源資料(data)維度需為1")


@check_array_and_process_nan_1d
@flatten_multi_dimension_array_1d
def interp1D_fast_layers(dx: float, xLoc: ndarray, data: ndarray):
    '''
    將等間距的1D坐標線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後一個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(xLoc) = (g,h)
         dim(output) = (a,b,c,d,e,g,h)
    (f) 為內插的維度，內插至(g,h)的位置
    (a,b,c,d,e)如時間、高度，沿這些維度重複內插(f)資料至(g,h)位置

    Parameters
    ----------
    dx : data 第0維(X)的直角網格資料間距
    xLoc : 要內插的目標點第0維(X)位置
    data : 1維的等間距直角網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後的陣列
    '''

    original_loc_shape = xLoc.shape
    xLoc = xLoc.reshape(-1)

    lenx = data.shape[1]
    N = xLoc.shape[0]
    layers = data.shape[0]

    if data.dtype != np.float64:
        data = data.astype(np.float64)

    if data.flags.f_contiguous != True:
        data = np.asfortranarray(data)

    output = np.empty([layers,N])
    output = np.asfortranarray(output)

    fastcompute.interp1D_fast_layers(ct.c_int(lenx),
                                     ct.c_int(layers),ct.c_double(dx),
                                    ct.c_int(N),
                                    xLoc.ctypes.data_as(cdouble_p),
                                    data.ctypes.data_as(cdouble_p),
                                    output.ctypes.data_as(cdouble_p))


    return output.reshape([data.shape[0]] + list(original_loc_shape))



@check_array_and_process_nan_1d_nonequal
@flatten_multi_dimension_array_1d
def interp1D_nonequal_layers(x_input: ndarray, x_output: ndarray, data: ndarray):
    '''
    將不等間距的1D坐標線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後一個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(x_output) = (g,h)
         dim(output) = (a,b,c,d,e,g,h)
    (f) 為內插的維度，內插至(g,h)的位置
    (a,b,c,d,e)如時間、高度，沿這些維度重複內插(e)資料至(g,h)位置

    Parameters
    ----------
    x_input : data 的原始格點位置
    x_output : 要內插的目標點位置
    data : 不等間距網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後的陣列
    '''
    original_loc_shape = x_output.shape
    x_output = x_output.reshape(-1)
    lenx = data.shape[1]
    N = x_output.shape[0]
    layers = data.shape[0]


    if data.dtype != np.float64:
        data = data.astype(np.float64)

    if x_input.dtype != np.float64:
        x_input = x_input.astype(np.float64)

    if data.flags.f_contiguous != True:
        data = np.asfortranarray(data)

    output = np.empty([layers,N])
    output = np.asfortranarray(output)

    fastcompute.interp1D_nonequal_layers(ct.c_int(lenx),x_input.ctypes.data_as(cdouble_p),
                                ct.c_int(layers), ct.c_int(N),
                                x_output.ctypes.data_as(cdouble_p),
                                data.ctypes.data_as(cdouble_p),
                                output.ctypes.data_as(cdouble_p))


    return output.reshape([data.shape[0]] + list(original_loc_shape))
