import os
from decorator import decorator
from meteotools.fileprocess import load_settings
from meteotools.grids import wrfout_grid, cylindrical_grid
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
        cyl = cylindrical_grid(f'./data/test_cyl_d_{curr_time}.nc',
                               interp_cpus=cpus)
        cyl.ncfile_readvars('u', 'v', 'w', 'f', 'p', 'T', 'Th',
                            'qv', 'qc', 'qr', 'qi', 'qs', 'qg')
        cyl.set_horizontal_location(0, 0)

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

        cyl.set_horizontal_location(0, 0)

        out_file_name = f'./data/test_cyl_d_{curr_time}.nc'
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
        cyl.ncfile_addvar('qc', ['vertical', 'tangential', 'radial'],
                          unit='kg/kg',
                          description='cloud water mixing ratio')
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


def plot_var_plan(cyl, var, levels, cmap='bwr'):
    fig, ax = plt.subplots()
    c = ax.contourf(cyl.xTR, cyl.yTR, var, cmap=cmap,
                    levels=levels, extend='both')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.show()


def plot_var_RZ(cyl, var, levels, cmap='bwr', z_plot=None, r_plot=None):
    fig, ax = plt.subplots()
    c = ax.contourf(cyl.r, cyl.z, var, cmap=cmap,
                    levels=levels, extend='both')
    if type(z_plot) is not type(None):
        ax.plot(z_plot, cyl.z, color='k')
    if type(r_plot) is not type(None):
        ax.plot(cyl.r, r_plot, color='k')

    fig.colorbar(c, ax=ax)
    plt.show()


def plot_r_1D(cyl, *varlist):
    fig, ax = plt.subplots()
    for idx, var in enumerate(varlist):
        ax.plot(cyl.r, var, label=idx)
    ax.legend()
    plt.show()


def plot_z_1D(cyl, *varlist):
    fig, ax = plt.subplots()
    for idx, var in enumerate(varlist):
        ax.plot(cyl.z, var, label=idx)
    ax.legend()
    plt.show()


@reset_cyl
def test_calc_correct():
    # 開始測試cyl functions
    cyl.calc_vr_vt()
    cyl.calc_wind_speed()
    # plot_var_plan(cyl, (cyl.v[2]**2 + cyl.u[2]**2)**0.5,
    #              np.arange(-0, 1.01, 0.1)*80, 'rainbow')
    # plot_var_plan(cyl, (cyl.vr[2]**2 + cyl.vt[2]**2)**0.5,
    #              np.arange(-0, 1.01, 0.1)*80, 'rainbow')
    # plot_var_plan(cyl, cyl.wind_speed[2],
    #              np.arange(-0, 1.01, 0.1)*80, 'rainbow')

    cyl.calc_axisymmetric_asymmetric('vt')
    cyl.find_rmw_and_speed()
    # plot_var_plan(cyl, cyl.asy_vt[2],
    #              np.arange(-1, 1.01, 0.1)*20, 'bwr')
    # plot_var_RZ(cyl, cyl.sym_vt,
    #            np.arange(-0, 1.01, 0.1)*70, 'rainbow', z_plot=cyl.RMW,)
    # plot_z_1D(cyl, cyl.vt_max_RMW, cyl.wspd_max_RMW)

    cyl.calc_agradient_force()
    # plot_var_plan(cyl, cyl.coriolis_force[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    # plot_var_plan(cyl, cyl.centrifugal_force[1],
    #              np.arange(-1, 1.01, 0.1)*1e-1, 'bwr')
    # plot_var_plan(cyl, cyl.radial_PG_force[1],
    #              np.arange(-1, 1.01, 0.1)*1e-1, 'bwr')
    # plot_var_plan(cyl, cyl.agradient_force[1],
    #              np.arange(-1, 1.01, 0.1)*1e-1, 'bwr')

    cyl.calc_kinetic_energy()
    # plot_var_plan(cyl, cyl.kinetic_energy[1],
    #              np.arange(-1, 1.01, 0.1)*3e3, 'bwr')

    cyl.calc_vorticity_3D()
    # plot_var_plan(cyl, cyl.zeta_r[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    # plot_var_plan(cyl, cyl.zeta_t[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    # plot_var_plan(cyl, cyl.zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    cyl.calc_abs_vorticity_3D()
    # plot_var_plan(cyl, cyl.abs_zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    cyl.calc_vorticity_z()
    # plot_var_plan(cyl, cyl.zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    cyl.calc_abs_vorticity_z()
    # plot_var_plan(cyl, cyl.abs_zeta_z[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    cyl.calc_divergence_hori()
    cyl.calc_divergence()
    # plot_var_plan(cyl, cyl.div_hori[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')
    # plot_var_plan(cyl, cyl.div[1],
    #              np.arange(-1, 1.01, 0.1)*1e-2, 'bwr')

    cyl.calc_density_potential_temperature()
    # plot_var_plan(cyl, cyl.Th_rho[1],
    #              np.arange(-1, 1.01, 0.1) * 10 + 305, 'rainbow')
    # plot_var_plan(cyl, cyl.Th[1],
    #              np.arange(-1, 1.01, 0.1) * 10 + 305, 'rainbow')
    # plot_var_plan(cyl, cyl.Th_rho[1] - cyl.Th[1],
    #              np.arange(-0, 1.01, 0.1) * 5, 'Reds')
    # plot_var_plan(cyl, cyl.qv[1],
    #              np.arange(-0, 1.01, 0.1) * 30/1000, 'Blues')
    # plot_var_plan(cyl, (cyl.qc+cyl.qr+cyl.qi+cyl.qs+cyl.qg)[1],
    #              np.arange(-0, 1.01, 0.1) * 12/1000, 'Blues')

    cyl.calc_PV()
    # plot_var_plan(cyl, cyl.PV[1],
    #              np.arange(-0, 1.01, 0.1) * 5e-5, 'Blues')

    cyl.calc_deformation()
    # plot_var_plan(cyl, cyl.shear_deform[1],
    #              np.arange(-1, 1.01, 0.1) * 1e-2, 'bwr')
    # plot_var_plan(cyl, cyl.stretch_deform[1],
    #              np.arange(-1, 1.01, 0.1) * 1e-2, 'bwr')

    cyl.calc_filamentation()
    # plot_var_plan(cyl, cyl.filamentation_time[5],
    #              np.arange(-0, 1.01, 0.1) * 3600, 'bwr')
    cyl.calc_axisymmetry('vt')
    #plot_var_RZ(cyl, cyl.axisymmetry_vt, np.arange(0, 1.01, 0.1), 'Reds')
    cyl.calc_axisymmetry('f')
    # print(cyl.axisymmetry_f)

    plot_var_plan(cyl, cyl.vt[2],
                  np.arange(-0, 1.01, 0.1)*80, 'rainbow')
    cyl.rotate_to_shear_direction('vt', 30, 'deg')
    plot_var_plan(cyl, cyl.shear_relative_vt[2],
                  np.arange(-0, 1.01, 0.1)*80, 'rainbow')


@reset_cyl
def test_calc_axisymmetric_asymmetric():
    assert ('sym_v' not in dir(cyl) and 'asy_v' not in dir(cyl)) == True
    cyl.calc_axisymmetric_asymmetric('v')
    assert ('sym_v' in dir(cyl) and 'asy_v' in dir(cyl)) == True
    del cyl.u
    assert check_exception(
        AttributeError, cyl.calc_axisymmetric_asymmetric, 'u')


@reset_cyl
def test_calc_vr_vt():
    cyl.calc_vr_vt()
    del cyl.u
    assert check_exception(AttributeError, cyl.calc_vr_vt)


@reset_cyl
def test_calc_fric_vr_vt():
    assert check_exception(AttributeError, cyl.calc_fric_vr_vt)


@reset_cyl
def test_calc_wind_speed():
    cyl.calc_wind_speed()
    del cyl.u
    assert check_exception(AttributeError, cyl.calc_wind_speed)


@reset_cyl
def test_find_rmw_and_speed():
    cyl.calc_wind_speed()
    del cyl.u
    assert check_exception(AttributeError, cyl.calc_wind_speed)


@reset_cyl
def test_calc_agradient_force():
    cyl.calc_agradient_force()
    del cyl.p
    assert check_exception(AttributeError, cyl.calc_agradient_force)


@reset_cyl
def test_calc_kinetic_energy():
    cyl.calc_kinetic_energy()
    del cyl.w
    assert check_exception(AttributeError, cyl.calc_kinetic_energy)


@reset_cyl
def test_calc_vorticity_3D():
    cyl.calc_vorticity_3D()
    del cyl.w
    assert check_exception(AttributeError, cyl.calc_vorticity_3D)


@reset_cyl
def test_calc_abs_vorticity_3D():
    cyl.calc_abs_vorticity_3D()
    del cyl.f
    assert check_exception(AttributeError, cyl.calc_abs_vorticity_3D)


@reset_cyl
def test_calc_vorticity_z():
    cyl.calc_vorticity_z()
    del cyl.v, cyl.vt
    assert check_exception(AttributeError, cyl.calc_vorticity_z)


@reset_cyl
def test_calc_abs_vorticity_z():
    cyl.calc_abs_vorticity_z()
    del cyl.f
    assert check_exception(AttributeError, cyl.calc_abs_vorticity_z)


@reset_cyl
def test_calc_divergence_hori():
    cyl.calc_divergence_hori()
    del cyl.v, cyl.vt
    assert check_exception(AttributeError, cyl.calc_divergence_hori)


@reset_cyl
def test_calc_divergence():
    cyl.calc_divergence()
    del cyl.w
    assert check_exception(AttributeError, cyl.calc_divergence)


@reset_cyl
def test_calc_density_factor():
    cyl.calc_density_factor()
    del cyl.qg
    cyl.calc_density_factor()
    del cyl.qs
    cyl.calc_density_factor()
    del cyl.qi
    cyl.calc_density_factor()
    del cyl.qr
    cyl.calc_density_factor()
    del cyl.qc
    cyl.calc_density_factor()
    del cyl.qv
    assert check_exception(AttributeError, cyl.calc_density_factor)


@reset_cyl
def test_calc_density_potential_temperature():
    cyl.calc_density_potential_temperature()


@reset_cyl
def test_calc_PV():
    cyl.calc_PV()


@reset_cyl
def test_calc_deformation():
    cyl.calc_deformation()


@reset_cyl
def test_calc_filamentation():
    cyl.calc_filamentation()


def test_dynamics():
    test_calc_vr_vt()
    test_calc_axisymmetric_asymmetric()
    test_calc_fric_vr_vt()
    test_calc_wind_speed()
    test_find_rmw_and_speed()
    test_calc_agradient_force()
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
    os.system('python default_settings_cyl_d.py')
    settings = load_settings('./settings/settings_cyl_d.json')
    cpus = settings['system']['calc_cpus']

    cyl = set_cyl(settings, cpus)
    test_calc_correct()
    test_dynamics()
