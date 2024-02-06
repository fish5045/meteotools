import os
from decorator import decorator
from meteotools.fileprocess import load_settings
from meteotools.grids import wrfout_grid, cartesian_grid
from meteotools.exceptions import DimensionError
import numpy as np
import matplotlib.pyplot as plt


def verification(ans, right_ans, tor=1e-13):
    return np.abs(ans - right_ans) < tor


def check_exception(exception, function, *args, **kargs):
    try:
        function(*args, **kargs)
        print('No error.')
    except exception:
        return 1


def reset_car(func):
    def wrapper(*args, **kargs):
        func(*args, **kargs)
        global car
        del car
        car = set_car(settings, cpus)
    return wrapper


def set_car(settings, cpus):
    wrfdata = wrfout_grid(settings['wrf'], interp_cpus=cpus)
    curr_time = wrfdata.wrfout_time


    try:
        car = cartesian_grid(f'./data/test_car_d_{curr_time}.nc',
                             interp_cpus=cpus)
        car.ncfile_readvars('u', 'v', 'w', 'f', 'p', 'T', 'Th',
                            'qv', 'qc', 'qr', 'qi', 'qs', 'qg')
        car.set_horizontal_location(0, 0)

    except:
        wrfdata.read_data(
            'ua', 'va', 'wa', 'MAPFAC_MX', 'MAPFAC_MY', 'F', 'z', 'pressure',
            'tk', 'theta', 'QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW',
            'QGRAUP')

        wrfdata.correct_map_scale()

        car = cartesian_grid(settings['cartesian_grid'],
                             interp_cpus=cpus)
        car.set_horizontal_location(
            wrfdata.center_xloc, wrfdata.center_yloc)

        car.set_data(wrfdata.dx, wrfdata.dy, wrfdata.z,
                     u=wrfdata.ua, v=wrfdata.va, w=wrfdata.wa,
                     f=wrfdata.F, p=wrfdata.pressure,
                     T=wrfdata.tk, Th=wrfdata.theta,
                     qv=wrfdata.QVAPOR, qc=wrfdata.QCLOUD,
                     qr=wrfdata.QRAIN, qi=wrfdata.QICE,
                     qs=wrfdata.QSNOW, qg=wrfdata.QGRAUP)

        car.set_horizontal_location(0, 0)

        out_file_name = f'./data/test_car_d_{curr_time}.nc'
        car.create_ncfile(out_file_name, wrfdata.time)
        car.ncfile_addvar('u', ['z', 'y', 'x'],
                          unit='m/s', description='WE wind')
        car.ncfile_addvar('v', ['z', 'y', 'x'],
                          unit='m/s', description='NS wind')
        car.ncfile_addvar('w', ['z', 'y', 'x'],
                          unit='m/s', description='vertical velocity')
        car.ncfile_addvar('f', ['y', 'x'],
                          unit='s-1', description='Coriolis parameters')
        car.ncfile_addvar('p', ['z', 'y', 'x'],
                          unit='Pa', description='pressure')
        car.ncfile_addvar('T', ['z', 'y', 'x'],
                          unit='K', description='temperature')
        car.ncfile_addvar('Th', ['z', 'y', 'x'],
                          unit='K', description='potential temperature')
        car.ncfile_addvar('qv', ['z', 'y', 'x'],
                          unit='kg/kg', description='vapor mixing ratio')
        car.ncfile_addvar('qc', ['z', 'y', 'x'],
                          unit='kg/kg',
                          description='cloud water mixing ratio')
        car.ncfile_addvar('qr', ['z', 'y', 'x'],
                          unit='kg/kg', description='rain water mixing ratio')
        car.ncfile_addvar('qi', ['z', 'y', 'x'],
                          unit='kg/kg', description='ice mixing ratio')
        car.ncfile_addvar('qs', ['z', 'y', 'x'],
                          unit='kg/kg', description='snow mixing ratio')
        car.ncfile_addvar('qg', ['z', 'y', 'x'],
                          unit='kg/kg', description='graupel mixing ratio')

        size = os.path.getsize(out_file_name)
        print(f'Complete to preprocess {curr_time}.')
        print(f'File: {out_file_name}')
        print(f'Size: {size/1024:.1f} KB')
    return car


def plot_var_plan(car, var, levels, cmap='bwr'):
    fig, ax = plt.subplots()
    c = ax.contourf(car.x, car.y, var, cmap=cmap,
                    levels=levels, extend='both')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.show()


def plot_var_RZ(car, var, levels, cmap='bwr', z_plot=None, r_plot=None):
    fig, ax = plt.subplots()
    c = ax.contourf(car.r, car.z, var, cmap=cmap,
                    levels=levels, extend='both')
    if type(z_plot) is not type(None):
        ax.plot(z_plot, car.z, color='k')
    if type(r_plot) is not type(None):
        ax.plot(car.r, r_plot, color='k')

    fig.colorbar(c, ax=ax)
    plt.show()


def plot_r_1D(car, *varlist):
    fig, ax = plt.subplots()
    for idx, var in enumerate(varlist):
        ax.plot(car.r, var, label=idx)
    ax.legend()
    plt.show()


def plot_z_1D(car, *varlist):
    fig, ax = plt.subplots()
    for idx, var in enumerate(varlist):
        ax.plot(car.z, var, label=idx)
    ax.legend()
    plt.show()


@reset_car
def test_calc_correct():
    # 開始測試car functions
    car.calc_wind_speed()
    # plot_var_plan(car, (car.v[2]**2 + car.u[2]**2)**0.5,
    #              np.arange(-0, 1.01, 0.1)*80, 'rainbow')
    # plot_var_plan(car, car.wind_speed[2],
    #              np.arange(-0, 1.01, 0.1)*80, 'rainbow')

    car.calc_kinetic_energy()
    # plot_var_plan(car, car.kinetic_energy[1],
    #              np.arange(-1, 1.01, 0.1)*3e3, 'bwr')

    car.calc_vorticity_3D()
    # plot_var_plan(car, car.zeta_r[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    # plot_var_plan(car, car.zeta_t[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    # plot_var_plan(car, car.zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    car.calc_abs_vorticity_3D()
    # plot_var_plan(car, car.abs_zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    car.calc_vorticity_z()
    # plot_var_plan(car, car.zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    car.calc_abs_vorticity_z()
    # plot_var_plan(car, car.abs_zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    car.calc_divergence_hori()
    car.calc_divergence()
    # plot_var_plan(car, car.div_hori[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    # plot_var_plan(car, car.div[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    car.calc_density_potential_temperature()
    # plot_var_plan(car, car.Th_rho[1],
    #              np.arange(-1, 1.01, 0.1) * 10 + 305, 'rainbow')
    # plot_var_plan(car, car.Th[1],
    #              np.arange(-1, 1.01, 0.1) * 10 + 305, 'rainbow')
    # plot_var_plan(car, car.Th_rho[1] - car.Th[1],
    #              np.arange(-0, 1.01, 0.1) * 5, 'Reds')
    # plot_var_plan(car, car.qv[1],
    #              np.arange(-0, 1.01, 0.1) * 30/1000, 'Blues')
    # plot_var_plan(car, (car.qc+car.qr+car.qi+car.qs+car.qg)[1],
    #              np.arange(-0, 1.01, 0.1) * 12/1000, 'Blues')

    car.calc_PV()
    # plot_var_plan(car, car.PV[1],
    #              np.arange(-0, 1.01, 0.1) * 5e-5, 'Blues')

    car.calc_deformation()
    # plot_var_plan(car, car.shear_deform[1],
    #              np.arange(-1, 1.01, 0.1) * 1e-2, 'bwr')
    # plot_var_plan(car, car.stretch_deform[1],
    #              np.arange(-1, 1.01, 0.1) * 1e-2, 'bwr')

    car.calc_filamentation()
    # plot_var_plan(car, car.filamentation_time[5],
    #              np.arange(-0, 1.01, 0.1) * 3600, 'bwr')

    # plot_var_plan(car, car.zeta_z[5],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    car.smooth1d('zeta_z', 0, passes=10)
    # plot_var_plan(car, car.smooth1d_zeta_z[5],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    car.smooth2d('zeta_z', (1, 2), passes=10)
    # plot_var_plan(car, car.smooth2d_zeta_z[5],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    car.smooth3d('zeta_z', passes=10)
    # plot_var_plan(car, car.smooth3d_zeta_z[5],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    car.calc_vertical_average('zeta_z', zmin=1000, zmax=3000)
    # plot_var_plan(car, car.zavg_zeta_z,
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    car.calc_vertical_average('PV')
    # plot_var_plan(car, car.zavg_PV,
    #              np.arange(-0, 1.01, 0.1) * 5e-5, 'Blues')

    car.calc_horizontal_average('zeta_z')
    #plot_z_1D(car, car.havg_zeta_z,)
    car.calc_horizontal_average('PV')
    #plot_z_1D(car, car.havg_PV,)
    car.calc_horizontal_average('f')
    # print(car.havg_f)


@reset_car
def test_calc_axisymmetric_asymmetric():
    assert ('sym_v' not in dir(car) and 'asy_v' not in dir(car)) == True
    car.calc_axisymmetric_asymmetric('v')
    assert ('sym_v' in dir(car) and 'asy_v' in dir(car)) == True
    assert check_exception(
        DimensionError, car.calc_axisymmetric_asymmetric, 'f')
    del car.u
    assert check_exception(
        AttributeError, car.calc_axisymmetric_asymmetric, 'u')


@reset_car
def test_calc_vr_vt():
    car.calc_vr_vt()
    del car.u
    assert check_exception(AttributeError, car.calc_vr_vt)


@reset_car
def test_calc_fric_vr_vt():
    assert check_exception(AttributeError, car.calc_fric_vr_vt)


@reset_car
def test_calc_wind_speed():
    car.calc_wind_speed()
    del car.u
    assert check_exception(AttributeError, car.calc_wind_speed)


@reset_car
def test_find_rmw_and_speed():
    car.calc_wind_speed()
    del car.u
    assert check_exception(AttributeError, car.calc_wind_speed)


@reset_car
def test_calc_agradient_force():
    car.calc_agradient_force()
    del car.p
    assert check_exception(AttributeError, car.calc_agradient_force)


@reset_car
def test_calc_kinetic_energy():
    car.calc_kinetic_energy()
    del car.w
    assert check_exception(AttributeError, car.calc_kinetic_energy)


@reset_car
def test_calc_vorticity_3D():
    car.calc_vorticity_3D()
    del car.w
    assert check_exception(AttributeError, car.calc_vorticity_3D)


@reset_car
def test_calc_abs_vorticity_3D():
    car.calc_abs_vorticity_3D()
    del car.f
    assert check_exception(AttributeError, car.calc_abs_vorticity_3D)


@reset_car
def test_calc_vorticity_z():
    car.calc_vorticity_z()
    del car.v
    assert check_exception(AttributeError, car.calc_vorticity_z)


@reset_car
def test_calc_abs_vorticity_z():
    car.calc_abs_vorticity_z()
    del car.f
    assert check_exception(AttributeError, car.calc_abs_vorticity_z)


@reset_car
def test_calc_divergence_hori():
    car.calc_divergence_hori()
    del car.v
    assert check_exception(AttributeError, car.calc_divergence_hori)


@reset_car
def test_calc_divergence():
    car.calc_divergence()
    del car.w
    assert check_exception(AttributeError, car.calc_divergence)


@reset_car
def test_calc_density_factor():
    car.calc_density_factor()
    del car.qg
    car.calc_density_factor()
    del car.qs
    car.calc_density_factor()
    del car.qi
    car.calc_density_factor()
    del car.qr
    car.calc_density_factor()
    del car.qc
    car.calc_density_factor()
    del car.qv
    assert check_exception(AttributeError, car.calc_density_factor)


@reset_car
def test_calc_density_potential_temperature():
    car.calc_density_potential_temperature()


@reset_car
def test_calc_PV():
    car.calc_PV()


@reset_car
def test_calc_deformation():
    car.calc_deformation()


@reset_car
def test_calc_filamentation():
    car.calc_filamentation()


def test_dynamics():
    test_calc_wind_speed()
    test_calc_kinetic_energy()
    test_calc_vorticity_3D()
    test_calc_abs_vorticity_3D()
    test_calc_vorticity_z()
    test_calc_abs_vorticity_z()
    test_calc_divergence_hori()
    test_calc_divergence()
    test_calc_density_factor()
    test_calc_density_potential_temperature()
    test_calc_PV()
    test_calc_deformation()
    test_calc_filamentation()


if __name__ == '__main__':
    os.system('python default_settings_car_d.py')
    settings = load_settings('./settings/settings_car_d.json')
    cpus = settings['system']['calc_cpus']

    car = set_car(settings, cpus)
    #test_calc_correct()
    #test_dynamics()
    #print(dir(car))
