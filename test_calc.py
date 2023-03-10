from meteotools.exceptions import LengthError, DimensionError, InputError
from meteotools.calc import cartesian2cylindrical, Make_cyclinder_coord, \
    FD2, FD2_back, FD2_front, FD2_2, FD2_2_back, FD2_2_front, difference_FFT, \
    Make_vertical_axis, Make_tangential_axis, Make_radial_axis, find_root2, \
    wswd_to_uv, uv_to_wswd, calc_Raverage, calc_Zaverage, calc_RZaverage

import numpy as np


def verification(ans, right_ans, tor=1e-13):
    return np.abs(ans - right_ans) < tor


def check_exception(exception, function, *args, **kargs):
    try:
        function(*args, **kargs)
        print('No error.')
    except exception:
        return 0


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
    # print(out[0, 0, 0])
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


def test_FD():
    a = make_data_3d(4, 5, 6)
    b = FD2_2_back(a, 2, 0)
    b = FD2_2(a, 2, 0)
    b = FD2_2_front(a, 2, 0)
    b = FD2_back(a, 2, 0)
    b = FD2(a, 2, 0)
    assert np.all(b == 0.5)
    b = FD2_front(a, 2, 0)
    assert np.all(np.isnan(b[-2:]))
    assert np.nansum(b) - 30 < 1e-13

    c = np.linspace(0, 2*np.pi, 361)[:-1]
    d = np.sin(c)
    b = difference_FFT(d, c[1]-c[0], 0)
    assert b.real[0] - 1 < 1e-13
    assert b.real[90] - 0 < 1e-13
    assert b.real[180] + 1 < 1e-13
    assert b.real[270] - 0 < 1e-13
    assert b.real[60] - 0.5 < 1e-13
    assert b.real[120] + 0.5 < 1e-13


def test_coordinates():
    z = Make_vertical_axis(0, 20, 0.1, 'km')
    assert np.sum(z - np.arange(0, 20000.01, 100.) < 1e-11) == 201
    t = Make_tangential_axis(0, 360, 1, 'deg')
    assert np.sum(t - np.linspace(0, 2*np.pi, 361)[:-1] < 1e-11) == 360
    r = Make_radial_axis(1, 300, 1, 'km')
    assert np.sum(r - np.arange(1000, 300000.01, 1000.) < 1e-10) == 300


def test_find_root2():
    a, b = find_root2(1, -2, -3)
    assert verification(a, 3) and verification(b, -1)
    a, b = find_root2(1, -2, 5)
    assert verification(a.real, 1) and verification(a.imag, 2)
    assert verification(b.real, 1) and verification(b.imag, -2)


def test_wswd_uv():
    u, v = wswd_to_uv(5, 180)
    assert verification(u, 0) and verification(v, 5)
    u, v = wswd_to_uv(5, 90)
    assert verification(u, -5) and verification(v, 0)
    ws, wd = uv_to_wswd(5, 5)
    assert verification(wd, 225) and verification(ws, np.sqrt(50))


def test_averages():
    a = np.eye(10)
    b = calc_Zaverage(a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)
    assert verification(np.sum(b), 1)
    assert b.shape[0] == 10
    b = calc_Zaverage(a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1, zmin=3)
    assert verification(np.sum(b), 1.)
    assert verification(b[5], 1/7)
    assert b.shape[0] == 10
    b = calc_Zaverage(a, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, zmin=3)
    assert verification(np.sum(b), 1.)
    assert verification(b[5], 1/8)
    assert b.shape[0] == 10
    assert (check_exception(ValueError, calc_Zaverage, a, [
            2, 1, 3, 4, 5, 6, 7, 8, 9, 10], 1, zmin=3)) == 0
    assert (check_exception(DimensionError, calc_Zaverage, a, [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 1, zmin=3)) == 0

    b = calc_Raverage(a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)
    assert verification(np.sum(b), 1)
    assert verification(b[1], 1/45)
    assert verification(b[4], 4/45)
    assert verification(b[7], 7/45)
    assert b.shape[0] == 10
    b = calc_Raverage(a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1, rmin=3)
    assert verification(b[1], 0/42)
    assert verification(b[4], 4/42)
    assert verification(b[7], 7/42)
    assert b.shape[0] == 10
    b = calc_Raverage(a, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, rmin=3)
    assert b.shape[0] == 10
    assert verification(b[1], 0/52)
    assert verification(b[4], 5/52)
    assert verification(b[7], 8/52)

    assert (check_exception(ValueError, calc_Raverage, a, [
            2, 1, 3, 4, 5, 6, 7, 8, 9, 10], 1, rmin=3)) == 0
    assert (check_exception(DimensionError, calc_Raverage, a, [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 1, rmin=3)) == 0

    z = np.arange(10)
    r = np.arange(10)
    b = calc_RZaverage(a, z, r)
    assert verification(b, 0.1)
    b = calc_RZaverage(a, z, r, rmin=3)
    assert verification(b, 8/80)
    b = calc_RZaverage(a, z, r, rmin=3, rmax=8, zmin=2, zmax=6, r_weight=False)
    assert verification(b, 4/30)
    b = calc_RZaverage(a, z, r, rmin=3, rmax=8, zmin=2, zmax=6)
    assert verification(b, (1*3+1*4+1*5+1*6)/(3+4+5+6+7+8)/5)
    assert (check_exception(DimensionError,
            calc_RZaverage, a, z, np.arange(8))) == 0
    z[3] = 20
    assert (check_exception(ValueError, calc_RZaverage, a, z, r)) == 0


if __name__ == '__main__':
    # test_Make_cyclinder_coord()
    # test_cartesian2cylindrical()
    # test_FD()
    # test_coordinates()
    # test_find_root2()
    # test_wswd_uv()
    # test_averages()
    # test_thermaldynamics() #未製作
    pass
