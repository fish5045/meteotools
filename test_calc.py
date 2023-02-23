from meteotools.exceptions import LengthError, DimensionError, InputError
from meteotools.calc import cartesian2cylindrical, Make_cyclinder_coord
import numpy as np


def make_r_coord(start, end, interval):
    return np.arange(start, end+0.0001, interval)


def make_z_coord(start, end, interval):
    return np.arange(start, end+0.0001, interval)


def make_theta_coord(number_of_grids):
    return np.linspace(0, np.pi*2, number_of_grids+1)[:-1]


def make_data_3d(n1, n2, n3):
    output = np.zeros([n1, n2, n3])
    for k in range(n1):
        for j in range(n2):
            for i in range(n3):
                output[k, j, i] = k + j + i
    return output


def test_Make_cyclinder_coord():
    r = make_r_coord(0, 100, 1)
    theta = make_theta_coord(360)
    xTR, yTR = Make_cyclinder_coord([0, 0], r, theta)
    xTR2, yTR2 = Make_cyclinder_coord([100, 0], r, theta)

    assert (xTR.shape == (360, 101))
    assert (np.abs(np.nanmax(yTR2 - yTR) - 100) < 1e-13)
    assert (np.abs(np.nanmin(yTR2 - yTR) - 100) < 1e-13)
    assert (np.sum(xTR2 - xTR) == 0)

    xTR3 = r.reshape(1, -1)*np.cos(theta.reshape(-1, 1))
    assert (np.abs(np.nanmax(xTR3 - xTR)) < 1e-13)


def test_cartesian2cylindrical():
    r = make_r_coord(0, 100, 1)
    theta = make_theta_coord(360)
    z = make_z_coord(10, 30, 1)
    data = make_data_3d(21, 251, 251)

    xTR, yTR = Make_cyclinder_coord([125, 125], r, theta)
    out = cartesian2cylindrical(1, 1, data, [125, 125], r, theta)
    #print(out[0, 0, 0])
    assert out[0, 0, 0] == 250.
    assert out[0, 0, 10] == 260.
    assert out[0, 0, 100] == 350.
    assert out[0, 90, 100] == 350.
    assert out[0, 180, 100] == 150.
    assert out[0, 270, 100] == 150.
    assert out[20, 0, 0] == 270.
    assert out[20, 0, 10] == 280.
    assert out[20, 0, 100] == 370.
    assert out[20, 90, 100] == 370.
    assert out[20, 180, 100] == 170.
    assert out[20, 270, 100] == 170.


if __name__ == '__main__':
    # test_Make_cyclinder_coord()
    # test_cartesian2cylindrical()
