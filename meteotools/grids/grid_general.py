from netCDF4 import Dataset
import numpy as np


import wrf


class gridsystem:
    '''
    網格系統的基底（陣列與網格資訊），包含metadata與從設定檔讀取metadata
    之後可以再衍生為直角網格、圓柱座標網格、wrf網格，可包含多個變數
    '''

    def set_wrfpython_cpus(self, cpus):
        wrf.omp_set_num_threads(cpus)

    def open_ncfile(self, filename):
        self.ncfilename = filename

        with Dataset(filename) as ncfile:
            for key, value in ncfile.dimensions.items():
                self.__readcoord(ncfile, key)

            for key, value in ncfile.__dict__.items():
                self.meta_list.append(key)
                try:
                    exec(f'self.{key} = {value}')
                except:
                    exec(f'self.{key} = {"value"}')

    def __init__(self, settings=None, wrf_prefix='wrfout_', interp_cpus=1):
        '''
        讀取網格設定
        '''
        self.wrf_prefix = wrf_prefix
        self.interp_cpus = interp_cpus
        self.set_wrfpython_cpus(self.interp_cpus)
        self.meta_list = []
        self.var_meta = dict()
        self.coords = dict()
        self.coord_ncname = dict()
        self.ncname_convert = dict(
            vertical='z', tangential='theta', radial='r')
        if settings != None:
            if type(settings) == dict:
                self.set_meta(settings)
                self.init_grid()   # 此段交由後續不同種網格做不同處理
            elif type(settings) == str:
                self.open_ncfile(settings)

    def set_meta(self, settings):
        for settingname in settings.keys():
            self.meta_list.append(settingname)

            if type(settings[settingname]) == str:
                text = f'{settings[settingname]}'
                exec(f"self.{settingname} = {text!r}")
            else:
                exec(f"self.{settingname} = {settings[settingname]}")

    def init_grid(self):
        pass

    def meta_convert_to_float(self, *metalist):
        for meta in metalist:
            exec(f'self.{meta} = float(self.{meta})')

    def meta_convert_to_int(self, *metalist):
        for meta in metalist:
            check_code = f'''
if self.{meta} != int(self.{meta}):
    print('Warning: wrfout setting "{meta}" is not a integer and will ignore the decimal.')
'''
            tmp = exec(check_code)
            exec(f'self.{meta} = int(self.{meta})')

    def set_var_meta(self, varname, **metadata):
        if self.var_meta.get(varname) == None:
            self.var_meta[varname] = dict()

        for key in metadata:
            exec(f"self.var_meta['{varname}']['{key}'] = metadata['{key}']")

    def set_coord_meta(self, name, ncname, coord, **metadata):
        self.coords[name] = coord
        self.coord_ncname[name] = ncname
        self.set_var_meta(name, **metadata)

    def __addvar(self, ncfile, varname, data, coord, dtype, metadata):
        var = ncfile.createVariable(varname, dtype, coord)
        for key in metadata:
            exec(f"var.{key} = metadata['{key}']")
        var[:] = data

    def __readcoord(self, ncfile, coordncname):
        try:
            coordname = self.ncname_convert[coordncname]
        except:
            coordname = coordncname
        tmp, metadata = self.__readvar(ncfile, coordncname)
        exec(f'self.{coordname} = tmp')
        self.set_coord_meta(coordname, coordncname, tmp, **metadata)

    def __readvar(self, ncfile, varname):
        var = ncfile.variables[varname]
        metadata = var.__dict__
        return np.array(var), metadata

    def ncfile_readvar(self, varname):
        with Dataset(self.ncfilename,) as ncfile:
            tmp, metadata = self.__readvar(ncfile, varname)
            exec(f'self.{varname} = tmp')
            self.var_meta[varname] = metadata
    
    def ncfile_readvars(self, *varlist):
        for var in varlist:
            self.ncfile_readvar(var)

    def ncfile_addvar(self, varname, coord, dtype='f4', **metadata):
        self.set_var_meta(varname, **metadata)
        data = eval(f'self.{varname}')

        try:
            with Dataset(self.ncfilename, 'a') as output:
                self.__addvar(output, varname, data, coord,
                              dtype, self.var_meta[varname])
        except:
            raise RuntimeError(
                'The nc file has not been set yet. Please run the method "create_ncfile" first.')

    def ncfile_addcoord(self, ncfile, coordname, dtype='f4'):
        ncfile.createDimension(
            self.coord_ncname[coordname],
            eval(f'self.N{coordname}'))
        self.__addvar(ncfile, self.coord_ncname[coordname], self.coords[coordname], [
                      self.coord_ncname[coordname]], 'f8', self.var_meta[coordname])

    def create_ncfile(self, filename, time=0):
        self.ncfilename = filename
        self.time = time
        self.Ntime = 1
        if self.time == 0:
            self.set_coord_meta(
                'time', 'time', self.time, unit='',
                description='No time information')
        else:
            self.set_coord_meta(
                'time', 'time', self.time, unit='s',
                description='Seconds from 1997/1/1 00:00:00')

        with Dataset(self.ncfilename, 'w', format='NETCDF4') as output:
            for settingname in self.meta_list:
                exec(f'output.{settingname} = self.{settingname}')

            for coordname, coord in self.coords.items():
                self.ncfile_addcoord(output, coordname)
