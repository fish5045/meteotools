from .check_arrays import check_array_and_process_nan_3d, \
    flatten_multi_dimension_array_3d
from meteotools.fastcompute import interp3d_fast, interp3d_fast_layers
from meteotools.exceptions import DimensionError
from numpy import ndarray


@check_array_and_process_nan_3d
def interp3D_fast(dx: float, dy: float, dz: float,
                  xLoc: ndarray, yLoc: ndarray, zLoc: ndarray,
                  data: ndarray) -> ndarray:
    '''
    將等間距的2D直角坐標雙線性內插至任意位置
    data的兩個維度網格間距必須相等

    Parameters
    ----------
    dx : data 第2維(X)的直角網格資料間距
    dy : data 第1維(Y)的直角網格資料間距
    dz : data 第0維(Z)的直角網格資料間距
    xLoc : 要內插的目標點第2維(X)位置
    yLoc : 要內插的目標點第1維(Y)位置
    zLoc : 要內插的目標點第0維(Z)位置
    data : 3維各方向等間距的網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後新位置的陣列，維度與xLoc、yLoc、zLoc相同

    '''

    if len(data.shape) == 3:
        output = interp3d_fast(
            dx, dy, dz,
            xLoc.reshape(-1), yLoc.reshape(-1), zLoc.reshape(-1),
            data)
        return output.reshape(xLoc.shape)

    else:
        raise DimensionError("來源資料(data)維度需為3")


@check_array_and_process_nan_3d
@flatten_multi_dimension_array_3d
def interp3D_fast_layers(dx: float, dy: float, dz: float,
                         xLoc: ndarray, yLoc: ndarray, zLoc: ndarray,
                         data: ndarray):
    '''
    將等間距的3D直角坐標雙線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後3個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(xLoc) = (g,h,i)
         dim(yLoc) = (g,h,i)
         dim(output) = (a,b,c,d,g,h,i)
    (e,f) 為內插的維度，內插至(g,h,i)的位置
    (a,b,c,d)如時間、高度，沿這些維度重複內插(e,f)資料至(g,h,i)位置

    Parameters
    ----------
    dx : data 第2維(X)的直角網格資料間距
    dy : data 第1維(Y)的直角網格資料間距
    dz : data 第0維(Z)的直角網格資料間距
    xLoc : 要內插的目標點第2維(X)位置
    yLoc : 要內插的目標點第1維(Y)位置
    zLoc : 要內插的目標點第0維(Z)位置
    data : 3維的等間距直角網格資料

    Raises
    ------
    DimensionError : 資料維度錯誤

    Returns
    -------
    內插後的陣列
    '''

    output = interp3d_fast_layers(
        dx, dy, dz,
        xLoc.reshape(-1), yLoc.reshape(-1), zLoc.reshape(-1),
        data)

    return output
