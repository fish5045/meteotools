from meteotools.exceptions import DimensionError
from decorator import decorator
import numpy as np


def check_dimension(variable, number_of_dim):
    if len(variable.shape) != number_of_dim:
        raise DimensionError(f'直角座標資料(car_data)須為{number_of_dim:d}維')


def check_monotonically_increasing(axis, axis_name):
    axis = np.array(axis)
    if np.min(axis[1:] - axis[:-1]) < 0:
        raise ValueError(f"{axis_name} axis must increase strictly.")


############### decorators ##########################
def pre_process_for_2Daverage(func):
    def wrapper(*args, **kargs):
        check_monotonically_increasing(args[1], 'z')
        check_monotonically_increasing(args[2], 'r')
        check_array_shape(args[1], args[0], 0)
        check_array_shape(args[2], args[0], 1)

        output = func(*args, **kargs)

        return output

    return wrapper


def check_array_shape(var_axis, var, axis):
    if var.shape[axis] != var_axis.shape[0]:
        raise DimensionError(
            f"The shape of axis ({var_axis.shape[0]:d}) and the shape of " +
            f"averaged dimension of var ({var.shape[axis]:d}) are not " +
            f"the same.")


@decorator
def pre_process_for_1Daverage(func, *args, **kargs):
    var_axis = np.array(args[1])
    check_monotonically_increasing(var_axis, 'input')
    check_array_shape(var_axis, args[0], args[2])
    var = np.moveaxis(args[0], args[2], -1)

    output = func(var, var_axis, *args[2:], **kargs)

    return output


def pre_post_process_for_differential(dest_axis):
    def pre_post_process(func):
        def wrapper(var, delta, axis):
            delta = float(delta)
            var = np.moveaxis(var, axis, dest_axis)

            output = func(var, delta, axis)

            output = np.moveaxis(output, dest_axis, axis)
            return output
        return wrapper
    return pre_post_process


def change_axis(dest_axis):
    @decorator
    def wrapper(func, *args, **kargs):
        var = np.moveaxis(args[0], args[1], dest_axis)

        output = func(var, *args[1:], **kargs)

        output = np.moveaxis(output, dest_axis, args[1])
        return output
    return wrapper
