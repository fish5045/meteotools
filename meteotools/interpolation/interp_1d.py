from .check_arrays import check_array_and_process_nan_1d, \
    check_array_and_process_nan_1d_nonequal, \
    flatten_multi_dimension_array_1d
from meteotools.fastcompute import interp1d_fast, interp1d, interp1d_fast_layers
from meteotools.exceptions import DimensionError
from numpy import ndarray, zeros


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
        output = interp1d_fast(dx, xLoc.reshape(-1), data)
        return output.reshape(xLoc.shape)
    else:
        raise DimensionError("來源資料(data)維度需為1")


@check_array_and_process_nan_1d_nonequal
def interp1D(x_input: ndarray, x_output: ndarray, data: ndarray) -> ndarray:
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
        output = interp1d(x_input.reshape(-1), x_output.reshape(-1), data)
        return output.reshape(x_output.shape)
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

    output = interp1d_fast_layers(dx, xLoc.reshape(-1), data)

    return output


@check_array_and_process_nan_1d_nonequal
@flatten_multi_dimension_array_1d
def interp1D_layers(x_input: ndarray, x_output: ndarray, data: ndarray):
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
    layers = data.shape[0]
    output = zeros([layers]+list(x_output.shape))
    for i in range(layers):
        output[i] = interp1d(
            x_input.reshape(-1),
            x_output.reshape(-1),
            data[i])

    return output
