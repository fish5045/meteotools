import numpy as np
from meteotools.exceptions import DimensionError, LengthError


def check_destination(destination: np.ndarray, dx: float, length: int,
                      dimension_name: str):
    '''
    檢查特定維度要內插的位置是否有超過資料範圍，超過會直接回報錯誤

    Parameters
    ----------
    destination : 要內插的位置
    dx : 網格間距
    length : 網格數
    dimension_name : 維度名稱

    Raises
    ------
    ValueError : 當內插位置超過資料範圍時，出現此錯誤
    '''
    destination_max = np.nanmax(destination)
    destination_min = np.nanmin(destination)

    if destination_max > dx * (length-1):
        raise ValueError(
            f"{dimension_name}位置最大值 {destination} "
            + f"超過網格範圍 0 ~ {dx * (length-1)}")
    if destination_min < 0:
        raise ValueError(
            f"{dimension_name}位置最小值 {destination_min} "
            + f"超過網格範圍 0 ~ {dx * (length-1)}")


def convert_to_ndarray(variable):
    return np.array(variable)


def check_array_shape(var1, var2):
    if var1.shape != var2.shape:
        raise DimensionError("xLoc與yLoc維度需相同")


def nan_to_numbers(variable):
    return np.where(np.isnan(variable), 1.7e308, variable)


def numbers_to_nan(variable):
    return np.where(variable == 1.7e308, np.nan, variable)


def check_destination_nonequal(destination: np.ndarray, origin: np.ndarray):
    '''
    檢查特定維度要內插的位置是否有超過資料範圍，超過會直接回報錯誤 (非固定網格間距)

    Parameters
    ----------
    destination : 要內插的位置
    origin : 原始網格各格點位置

    Raises
    ------
    ValueError : 當內插位置超過資料範圍時，出現此錯誤
    '''
    # 檢查x_output內插目標位置是否有超過資料範圍
    destination_max = np.nanmax(destination)
    origin_max = np.nanmax(origin)
    destination_min = np.nanmin(destination)
    origin_min = np.nanmin(origin)
    if destination_max > origin_max:
        raise ValueError(
            f'x_output 位置最大值 {destination_max} 超過資料範圍 {origin_min} ~ {origin_max}')
    if destination_min < origin_min:
        raise ValueError(
            f'x_output 位置最小值 {destination_min} 超過資料範圍 {origin_min} ~ {origin_max}')


def check_monotonically(variable):
    difference = variable[1:] - variable[:-1]
    if (np.all(difference > 0) or np.all(difference < 0)) == False:
        raise ValueError('x_input位置陣列須為單調遞增或單調遞減')




####################### decorators #######################################

def check_array_and_process_nan_1d_nonequal(func):
    def wrapper(x_input, x_output, data):
        x_output = convert_to_ndarray(x_output)
        x_input = convert_to_ndarray(x_input)
        check_monotonically(x_input)
        check_destination_nonequal(x_output, x_input)
        x_output = nan_to_numbers(x_output)

        output = func(x_input, x_output, data)
        output = numbers_to_nan(output)
        return output
    return wrapper


def check_array_and_process_nan_1d(func):
    def wrapper(dx, xLoc, data):
        xLoc = convert_to_ndarray(xLoc)
        check_destination(xLoc, dx, data.shape[-1], 'X')
        xLoc2 = nan_to_numbers(xLoc)

        output = func(dx,  xLoc2, data)

        output = numbers_to_nan(output)
        return output
    return wrapper


def check_array_and_process_nan_2d(func):
    def wrapper(dx, dy, xLoc, yLoc, data):
        xLoc = convert_to_ndarray(xLoc)
        yLoc = convert_to_ndarray(yLoc)
        check_array_shape(xLoc, yLoc)
        check_destination(xLoc, dx, data.shape[-1], 'X')
        check_destination(yLoc, dy, data.shape[-2], 'Y')
        xLoc2 = nan_to_numbers(xLoc)
        yLoc2 = nan_to_numbers(yLoc)

        output = func(dx, dy, xLoc2, yLoc2, data)

        output = numbers_to_nan(output)
        return output
    return wrapper


def check_array_and_process_nan_3d(func):
    def wrapper(dx, dy, dz, xLoc, yLoc, zLoc, data):
        xLoc = convert_to_ndarray(xLoc)
        yLoc = convert_to_ndarray(yLoc)
        zLoc = convert_to_ndarray(zLoc)
        check_array_shape(xLoc, yLoc)
        check_array_shape(zLoc, yLoc)
        check_destination(xLoc, dx, data.shape[-1], 'X')
        check_destination(yLoc, dy, data.shape[-2], 'Y')
        check_destination(zLoc, dz, data.shape[-3], 'Z')
        xLoc2 = nan_to_numbers(xLoc)
        yLoc2 = nan_to_numbers(yLoc)
        zLoc2 = nan_to_numbers(zLoc)

        output = func(dx, dy, dz, xLoc2, yLoc2, zLoc2, data)

        output = numbers_to_nan(output)
        return output
    return wrapper


def flatten_multi_dimension_array_1d(func):
    def wrapper(dx, xLoc, data):
        output_dimension = data.shape[:-1] + xLoc.shape
        layers = int(np.cumprod(data.shape[:-1])[-1])
        data = data.reshape([layers]+list(data.shape[-1:]))

        output = func(dx, xLoc, data)

        output = output.reshape(output_dimension)
        return output
    return wrapper


def flatten_multi_dimension_array_2d(func):
    def wrapper(dx, dy, xLoc, yLoc, data):
        output_dimension = data.shape[:-2] + xLoc.shape
        layers = int(np.cumprod(data.shape[:-2])[-1])
        data = data.reshape([layers]+list(data.shape[-2:]))

        output = func(dx, dy, xLoc, yLoc, data)

        output = output.reshape(output_dimension)
        return output
    return wrapper


def flatten_multi_dimension_array_3d(func):
    def wrapper(dx, dy, dz, xLoc, yLoc, zLoc, data):
        output_dimension = data.shape[:-3] + xLoc.shape
        layers = int(np.cumprod(data.shape[:-3])[-1])
        data = data.reshape([layers]+list(data.shape[-3:]))

        output = func(dx, dy, dz, xLoc, yLoc, zLoc, data)

        output = output.reshape(output_dimension)
        return output
    return wrapper
