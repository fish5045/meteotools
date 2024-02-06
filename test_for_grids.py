from meteotools.grids import wrfout_grid, cartesian_grid
import os
from meteotools.fileprocess import load_settings


def set_car(settings, cpus):
    wrfdata = wrfout_grid(settings['wrf'], interp_cpus=cpus)
    curr_time = wrfdata.wrfout_time




    try:
        0/0
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

        import numpy as np
        car.set_data(wrfdata.dx, wrfdata.dy, wrfdata.z,
                     u=wrfdata.ua, v=wrfdata.va, w=wrfdata.wa,
                     f=wrfdata.F, p=wrfdata.pressure,
                     T=wrfdata.tk, Th=wrfdata.theta,
                     qv=wrfdata.QVAPOR, qc=wrfdata.QCLOUD,
                     qr=wrfdata.QRAIN, qi=wrfdata.QICE,
                     qs=wrfdata.QSNOW, qg=wrfdata.QGRAUP,
                     hgt = np.zeros([810,810])+50)
        print(car.qv[0,10,10],car.qv[1,10,10],car.qv[2,10,10])

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


if __name__ == '__main__':
    os.system('python default_settings_car_d.py')
    settings = load_settings('./settings/settings_car_d.json')
    cpus = settings['system']['calc_cpus']

    car = set_car(settings, cpus)






