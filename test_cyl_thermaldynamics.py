import os
from decorator import decorator
from meteotools.fileprocess import load_settings
from meteotools.grids import wrfout_grid, cylindrical_grid
import numpy as np


def verification(ans, right_ans, tor=1e-13):
    return np.abs(ans - right_ans) < tor


def check_exception(exception, function, *args, **kargs):
    try:
        function(*args, **kargs)
        print('No error.')
    except exception:
        return 1


def reset_cyl(func):
    def wrapper(*args, **kargs):
        func(*args, **kargs)
        global cyl
        del cyl
        cyl = set_cyl(settings, cpus)
    return wrapper


def set_cyl(settings, cpus):
    wrfdata = wrfout_grid(settings['wrf'], interp_cpus=cpus)
    curr_time = wrfdata.wrfout_time

    try:
        cyl = cylindrical_grid(f'./data/test_cyl_td_{curr_time}.nc',
                               interp_cpus=cpus)
        cyl.ncfile_readvars('u', 'v', 'w', 'f', 'p', 'T', 'Th',
                            'qv', 'qc', 'qr', 'qi', 'qs', 'qg')
    except:
        wrfdata.read_data(
            'ua', 'va', 'wa', 'MAPFAC_MX', 'MAPFAC_MY', 'F', 'z', 'pressure',
            'tk', 'theta', 'QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW',
            'QGRAUP')

        wrfdata.correct_map_scale()

        cyl = cylindrical_grid(settings['cylindrical_grid'],
                               interp_cpus=cpus)
        cyl.set_horizontal_location(
            wrfdata.center_xloc, wrfdata.center_yloc)

        cyl.set_data(wrfdata.dx, wrfdata.dy, wrfdata.z,
                     u=wrfdata.ua, v=wrfdata.va, w=wrfdata.wa,
                     f=wrfdata.F, p=wrfdata.pressure,
                     T=wrfdata.tk, Th=wrfdata.theta,
                     qv=wrfdata.QVAPOR, qc=wrfdata.QCLOUD,
                     qr=wrfdata.QRAIN, qi=wrfdata.QICE,
                     qs=wrfdata.QSNOW, qg=wrfdata.QGRAUP)

        out_file_name = f'./data/test_cyl_td_{curr_time}.nc'
        cyl.create_ncfile(out_file_name, wrfdata.time)
        cyl.ncfile_addvar('u', ['vertical', 'tangential', 'radial'],
                          unit='m/s', description='storm-relative WE wind')
        cyl.ncfile_addvar('v', ['vertical', 'tangential', 'radial'],
                          unit='m/s', description='storm-relative NS wind')
        cyl.ncfile_addvar('w', ['vertical', 'tangential', 'radial'],
                          unit='m/s', description='vertical velocity')
        cyl.ncfile_addvar('f', ['tangential', 'radial'],
                          unit='s-1', description='Coriolis parameters')
        cyl.ncfile_addvar('p', ['vertical', 'tangential', 'radial'],
                          unit='Pa', description='pressure')
        cyl.ncfile_addvar('T', ['vertical', 'tangential', 'radial'],
                          unit='K', description='temperature')
        cyl.ncfile_addvar('Th', ['vertical', 'tangential', 'radial'],
                          unit='K', description='potential temperature')
        cyl.ncfile_addvar('qv', ['vertical', 'tangential', 'radial'],
                          unit='kg/kg', description='vapor mixing ratio')
        cyl.ncfile_addvar(
            'qc', ['vertical', 'tangential', 'radial'],
            unit='kg/kg', description='cloud water mixing ratio')
        cyl.ncfile_addvar('qr', ['vertical', 'tangential', 'radial'],
                          unit='kg/kg', description='rain water mixing ratio')
        cyl.ncfile_addvar('qi', ['vertical', 'tangential', 'radial'],
                          unit='kg/kg', description='ice mixing ratio')
        cyl.ncfile_addvar('qs', ['vertical', 'tangential', 'radial'],
                          unit='kg/kg', description='snow mixing ratio')
        cyl.ncfile_addvar('qg', ['vertical', 'tangential', 'radial'],
                          unit='kg/kg', description='graupel mixing ratio')

        size = os.path.getsize(out_file_name)
        print(f'Complete to preprocess {curr_time}.')
        print(f'File: {out_file_name}')
        print(f'Size: {size/1024:.1f} KB')
    return cyl


@reset_cyl
def test_calc_theta():
    del cyl.Th
    cyl.calc_theta()
    del cyl.T
    assert check_exception(AttributeError, cyl.calc_theta)
    cyl.calc_T()
    del cyl.p
    assert check_exception(AttributeError, cyl.calc_theta)


@reset_cyl
def test_calc_T():
    del cyl.T
    cyl.calc_T()
    del cyl.Th
    assert check_exception(AttributeError, cyl.calc_T)
    cyl.calc_theta()
    del cyl.p
    assert check_exception(AttributeError, cyl.calc_T)
    return


@reset_cyl
def test_calc_saturated_vapor():
    cyl.calc_saturated_vapor()
    del cyl.T
    cyl.calc_saturated_vapor()


@reset_cyl
def test_calc_saturated_qv():
    cyl.calc_saturated_qv()
    del cyl.T
    cyl.calc_saturated_qv()
    del cyl.p
    assert check_exception(AttributeError, cyl.calc_saturated_qv)


@reset_cyl
def test_calc_vapor():
    cyl.calc_vapor()
    cyl.calc_RH()
    cyl.calc_Td()
    del cyl.qv
    cyl.calc_vapor()
    del cyl.Td
    cyl.calc_vapor()
    del cyl.T
    cyl.calc_vapor()
    del cyl.Th
    del cyl.T
    assert check_exception(AttributeError, cyl.calc_vapor)


@reset_cyl
def test_calc_Td():
    cyl.calc_vapor()
    del cyl.qv
    assert check_exception(AttributeError, cyl.calc_vapor)


@reset_cyl
def test_calc_qv():
    cyl.calc_qv()
    del cyl.p
    assert check_exception(AttributeError, cyl.calc_qv)


@reset_cyl
def test_calc_theta_es():
    cyl.calc_theta_es()
    del cyl.T
    cyl.calc_theta_es()
    del cyl.p
    assert check_exception(AttributeError, cyl.calc_qv)


@reset_cyl
def test_calc_theta_v():
    cyl.calc_theta_v()
    del cyl.qr
    cyl.calc_theta_v()
    del cyl.qc
    cyl.calc_theta_v()
    del cyl.T
    cyl.calc_theta_v()
    del cyl.qv
    assert check_exception(AttributeError, cyl.calc_theta_v)


@reset_cyl
def test_calc_Tv():
    cyl.calc_Tv()
    del cyl.T
    cyl.calc_Tv()
    del cyl.qv
    assert check_exception(AttributeError, cyl.calc_Tv)


@reset_cyl
def test_calc_rho():
    cyl.calc_rho()
    del cyl.p
    assert check_exception(AttributeError, cyl.calc_rho)


@reset_cyl
def test_calc_Tc():
    cyl.calc_Tc()
    del cyl.T
    cyl.calc_Tc()
    del cyl.qv
    assert check_exception(AttributeError, cyl.calc_Tc)


@reset_cyl
def test_calc_theta_e():
    cyl.calc_theta_e()
    del cyl.Th
    cyl.calc_theta_e()
    del cyl.qv
    assert check_exception(AttributeError, cyl.calc_theta_e)


@reset_cyl
def test_calc_moist_dTdz():
    cyl.calc_moist_dTdz()
    del cyl.T
    cyl.calc_moist_dTdz()
    cyl.calc_saturated_qv()
    cyl.calc_moist_dTdz()


@reset_cyl
def test_calc_dry_dTdz():
    cyl.calc_dry_dTdz()


@reset_cyl
def test_calc_RH():
    cyl.calc_RH()
    del cyl.T
    cyl.calc_RH()
    del cyl.vapor
    cyl.calc_RH()


def test_calc_thermaldynamics():
    test_calc_theta()
    test_calc_T()
    test_calc_saturated_vapor()
    test_calc_saturated_qv()
    test_calc_vapor()
    test_calc_Td()
    test_calc_qv()
    test_calc_theta_es()
    test_calc_theta_v()
    test_calc_Tv()
    test_calc_rho()
    test_calc_Tc()
    test_calc_theta_e()
    test_calc_moist_dTdz()
    test_calc_dry_dTdz()
    test_calc_RH()


if __name__ == '__main__':
    os.system('python default_settings_cyl_td.py')
    settings = load_settings('./settings/settings_cyl_td.json')
    cpus = settings['system']['calc_cpus']

    cyl = set_cyl(settings, cpus)

    # 開始測試cyl functions
    test_calc_thermaldynamics()
