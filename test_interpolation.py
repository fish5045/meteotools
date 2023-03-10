from meteotools.exceptions import LengthError, DimensionError, InputError
from meteotools.interpolation import interp2D_fast, interp3D_fast, interp1D_fast, \
    interp1D_fast_layers, interp2D_fast_layers, interp3D_fast_layers, interp1D, interp1D_layers
import numpy as np
# from fastcompute import interp2D_fast


def check_exception(exception, function, *args, **kargs):
    try:
        function(*args, **kargs)
        print('No error.')
    except exception:
        return 0


def array_generator_1d(n):
    out = np.zeros([n])
    for i in range(n):
        out[i] = i
    return out


def array_generator_2d(n):
    out = np.zeros([n, n])
    for j in range(n):
        for i in range(n):
            out[j, i] = j + i
    return out


def array_generator_3d(n):
    out = np.zeros([n, n, n])
    for k in range(n):
        for j in range(n):
            for i in range(n):
                out[k, j, i] = k + j + i
    return out


def array_generator_1d_layers(n1, n2, n3):
    out = np.zeros([n1, n2, n3])
    for k in range(n1):
        for j in range(n2):
            for i in range(n3):
                out[k, j, i] = k + j + i
    return out


def array_generator_2d_layers(n1, n2, n3, n4):
    out = np.zeros([n1, n2, n3, n4])
    for m in range(n1):
        for k in range(n2):
            for j in range(n3):
                for i in range(n4):
                    out[m, k, j, i] = m + k + j + i
    return out


def array_generator_3d_layers(n1, n2, n3, n4, n5):
    out = np.zeros([n1, n2, n3, n4, n5])
    for n in range(n1):
        for m in range(n2):
            for k in range(n3):
                for j in range(n4):
                    for i in range(n5):
                        out[n, m, k, j, i] = n + m + k + j + i
    return out


def test_interp1D_fast():
    a = array_generator_1d(5)
    x = [0.5, 2.5, 3.5]

    b = interp1D_fast(1, x, a)

    assert (np.nanmax(np.abs(b - np.array([0.5, 2.5, 3.5]))) < 1e-13)

    a[1] = np.nan
    b = interp1D_fast(1, x, a)
    assert (np.nanmax(np.abs(b - np.array([np.nan, 2.5, 3.5]))) < 1e-13)
    assert np.isnan(b[0])

    assert check_exception(DimensionError, interp1D_fast,
                           1, x, np.stack([a]*2, axis=0)) == 0

    x = [1.7, 2.5, np.nan]
    b = interp1D_fast(1, x, a)
    assert (np.nanmax(np.abs(b - np.array([1.7, 2.5, np.nan]))) < 1e-13)
    assert np.isnan(b[2])

    x = [1.7, 7.5]
    assert check_exception(ValueError, interp1D_fast, 1, x, a) == 0


def test_interp1D():
    a = array_generator_1d(5)
    xin = [0, 2, 4, 5, 6]
    x = [0.5, 2.5, 5.5]

    b = interp1D(xin, x, a)

    assert (np.nanmax(np.abs(b - np.array([0.25, 1.25, 3.5]))) < 1e-13)

    a[1] = np.nan
    b = interp1D(xin, x, a)
    assert (np.nanmax(np.abs(b - np.array([np.nan, 1.25, 3.5]))) < 1e-13)
    assert np.isnan(b[0])

    assert check_exception(DimensionError, interp1D,
                           xin, x, np.stack([a]*2, axis=0)) == 0

    x = [1.7, 2.5, np.nan]
    a[1] = 1
    b = interp1D(xin, x, a)
    assert (np.nanmax(np.abs(b - np.array([0.85, 1.25, np.nan]))) < 1e-13)
    assert np.isnan(b[2])

    x = [1.7, 7.5]
    assert check_exception(ValueError, interp1D, xin, x, a) == 0


def test_interp1D_layers():
    a = array_generator_1d_layers(2, 7, 5)
    xin = [0, 2, 4, 5, 6]
    x = [0.5, 2.5, 5.5]

    b = interp1D_layers(xin, x, a)
    expect_answer = np.zeros([2, 7, 3])
    base = np.array([0.25, 1.25, 3.5])
    for k in range(2):
        for j in range(7):
            for i in range(3):
                expect_answer[k, j, i] = base[i] + j + k

    assert (np.nanmax(np.abs(b - expect_answer)) < 1e-13)

    a[1, 1, 1] = np.nan
    b = interp1D_layers(xin, x, a)
    assert (np.nanmax(np.abs(b - expect_answer)) < 1e-13)
    assert np.isnan(b[1, 1, 0])
    assert np.isnan(b[1, 1, 1])

    x = [0.5, 2.5, np.nan]
    a[1, 1, 1] = 3
    b = interp1D_layers(xin, x, a)
    assert (np.nanmax(np.abs(b - expect_answer)) < 1e-13)
    assert (np.sum(np.isnan(b[:, :, 2])) == 14)

    x = [1.7, 7.5]
    assert check_exception(ValueError, interp1D, xin, x, a) == 0


def test_interp2D_fast():
    a = array_generator_2d(5)
    x = [1.7, 2.5, 3.5]
    y = [1.2, 2.3, 3.4]

    b = interp2D_fast(1, 1, x, y, a)
    assert (np.nanmax(np.abs(b - np.array([2.9, 4.8, 6.9]))) < 1e-13)

    a[1, 1] = np.nan
    b = interp2D_fast(1, 1, x, y, a)
    assert (np.nanmax(np.abs(b - np.array([np.nan, 4.8, 6.9]))) < 1e-13)
    assert np.isnan(b[0])

    x = [1.7, 2.5, np.nan]
    b = interp2D_fast(1, 1, x, y, a)
    assert (np.nanmax(np.abs(b - np.array([2.9, 4.8, np.nan]))) < 1e-13)
    assert np.isnan(b[2])

    assert check_exception(DimensionError, interp2D_fast,
                           1, 1, x, y, np.stack([a]*2, axis=0)) == 0

    x = [1.7, 2.5]
    assert check_exception(DimensionError, interp2D_fast, 1, 1, x, y, a) == 0

    x = [1.7, 7.5, 3.5]
    assert check_exception(ValueError, interp2D_fast, 1, 1, x, y, a) == 0

    x = [[1.7, 2.5, 3.5]]
    assert check_exception(DimensionError, interp2D_fast, 1, 1, x, y, a) == 0


def test_interp3D_fast():
    a = array_generator_3d(5)
    x = [0, 2.5, 3.5]
    y = [0, 2.3, 3.4]
    z = [0.1, 1.8, 2.9]

    b = interp3D_fast(1, 1, 1, x, y, z, a)
    assert (np.nanmax(np.abs(b - np.array([0.1, 6.6, 9.8]))) < 1e-13)

    a[1, 1, 1] = np.nan
    b = interp3D_fast(1, 1, 1, x, y, z, a)
    assert (np.nanmax(np.abs(b - np.array([np.nan, 6.6, 9.8]))) < 1e-13)
    assert np.isnan(b[0])

    x = [2.1, 2.5, np.nan]
    b = interp3D_fast(1, 1, 1, x, y, z, a)

    assert (np.nanmax(np.abs(b - np.array([2.2, 6.6, np.nan]))) < 1e-13)
    assert np.isnan(b[2])

    assert check_exception(DimensionError, interp3D_fast,
                           1, 1, 1, x, y, z, np.stack([a]*2, axis=0)) == 0

    x = [1.7, 2.5]
    assert check_exception(DimensionError, interp3D_fast,
                           1, 1, 1, x, y, z, a) == 0

    x = [1.7, 7.5, 3.5]
    assert check_exception(ValueError, interp3D_fast, 1, 1, 1, x, y, z, a) == 0

    x = [[1.7, 2.5, 3.5]]
    assert check_exception(DimensionError, interp3D_fast,
                           1, 1, 1, x, y, z, a) == 0


def test_interp1D_fast_layers():
    a = array_generator_1d_layers(2, 7, 5)
    x = [0.5, 2.5, 3.5]

    b = interp1D_fast_layers(1, x, a)
    expect_answer = np.zeros([2, 7, 3])
    for k in range(2):
        for j in range(7):
            for i in range(3):
                expect_answer[k, j, i] = x[i] + j + k

    assert (np.nanmax(np.abs(expect_answer-b)) < 1e-13)
    assert (b.shape == (2, 7, 3))

    a[1, 1, 1] = np.nan
    b = interp1D_fast_layers(1, x, a)
    assert(np.isnan(b[1, 1, 0]))

    x = [0.5, 7.5, 3.5]
    assert check_exception(ValueError, interp1D_fast, 1, x, a) == 0


def test_interp2D_fast_layers():
    a = array_generator_2d_layers(2, 3, 7, 5)
    x = [0.5, 2.5, 3.5]
    y = [1.9, 2.3, 3.4]
    xx, yy = np.meshgrid(x, y)

    b = interp2D_fast_layers(1, 1, x, y, a)
    expect_answer = np.zeros([2, 3, 3])
    for n in range(2):
        for m in range(3):
            for i in range(3):
                expect_answer[n, m, i] = x[i] + y[i] + n + m

    assert (np.nanmax(np.abs(expect_answer-b)) < 1e-13)
    assert (b.shape == (2, 3, 3))

    b = interp2D_fast_layers(1, 1, xx, yy, a)
    expect_answer = np.zeros([2, 3, 3, 3])
    for m in range(2):
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    expect_answer[m, k, j, i] = xx[j, i] + yy[j, i] + k + m

    assert (np.nanmax(np.abs(expect_answer-b)) < 1e-13)
    assert (b.shape == (2, 3, 3, 3))


def test_interp3D_fast_layers():
    a = array_generator_3d_layers(2, 4, 5, 7, 5)
    x = [0.5, 2.5, 3.5]
    y = [1.9, 2.3, 3.4]
    z = [0.3, 1.8, 2.9]
    xx, yy, zz = np.meshgrid(x, y, z)

    b = interp3D_fast_layers(1, 1, 1, x, y, z, a)
    expect_answer = np.zeros([2, 4, 3])
    for n in range(2):
        for m in range(4):
            for i in range(3):
                expect_answer[n, m, i] = x[i] + y[i] + z[i] + n + m
    assert (np.nanmax(np.abs(expect_answer-b)) < 1e-13)
    assert (b.shape == (2, 4, 3))

    b = interp3D_fast_layers(1, 1, 1, xx, yy, zz, a)
    expect_answer = np.zeros([2, 4, 3, 3, 3])
    for n in range(2):
        for m in range(4):
            for k in range(3):
                for j in range(3):
                    for i in range(3):
                        expect_answer[n, m, k, j, i] = xx[k, j,
                                                          i] + yy[k, j, i] + zz[k, j, i] + n + m

    assert (np.nanmax(np.abs(expect_answer-b)) < 1e-13)
    assert (b.shape == (2, 4, 3, 3, 3))


if __name__ == '__main__':
    test_interp2D_fast()
    test_interp3D_fast()
    test_interp1D_fast()
    test_interp1D()
    test_interp1D_layers()
    test_interp1D_fast_layers()
    test_interp2D_fast_layers()
    test_interp3D_fast_layers()
