from meteotools.exceptions import DimensionError


def check_dimension(variable, number_of_dim):
    if len(variable.shape) != number_of_dim:
        raise DimensionError(f'直角座標資料(car_data)須為{number_of_dim:d}維')
