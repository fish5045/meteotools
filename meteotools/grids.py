import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from scipy.interpolate import splev, splrep
import wrf
import time as tm
import multiprocessing as mp
import json

from meteotools import timer
from meteotools.fileprocess import Get_wrfout_time, Get_time_sec
from meteotools.fastcompute import interp2D_fast_layers, interp2D_fast, Make_cyclinder_coord

interp_cpus = 8


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

    def __init__(self, settings=None, wrf_prefix='wrfout_'):
        '''
        讀取網格設定
        '''
        self.wrf_prefix = wrf_prefix
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
                text = f'"{settings[settingname]}"'
                exec(f"self.{settingname} = {text}")
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


# 此為wrfout的網格座標系統
class wrfout(gridsystem):
    '''
    wrfout 的網格系統，3維網格(z, y, x)，x, y為等間距的水平座標，z為eta座標
    '''

    def init_grid(self):
        self.offset = 0

        # 設定由josn檔案來，其數字不分整數或浮點數，由這兩個function強制區分
        self.meta_convert_to_float()
        self.meta_convert_to_int('start_yr', 'start_mo', 'start_dy',
                                 'start_hr', 'start_mi', 'start_sc',
                                 'end_yr', 'end_mo', 'end_dy',
                                 'end_hr', 'end_mi', 'end_sc',
                                 'domain')

        # 由設定檔得到的時間資訊，生成對應的wrfout檔名
        self.set_wrfout_fname()

    def set_wrfout_fname(self):
        self.wrfout_time = Get_wrfout_time(self.start_yr, self.start_mo,
                                           self.start_dy, self.start_hr,
                                           self.start_mi, self.start_sc,
                                           self.offset)
        self.time = Get_time_sec(self.start_yr, self.start_mo,
                                 self.start_dy, self.start_hr,
                                 self.start_mi, self.start_sc,
                                 self.offset)
        self.wrfout_fname = \
            f"{self.wrfout_dir}/{self.wrf_prefix}d{self.domain:02d}_{self.wrfout_time}"

    def set_offset(self, second):
        '''
        設定相對於設定wrfout時間，要取偏移多少時間的wrfout
        (可用於取local tendency需要取的前後時間點的wrfout)
        '''
        self.offset = second
        self.set_wrfout_fname()

    def read_data(self, *varlist):
        '''
        讀取wrfout資料，可給定wrf-python的getvar支援的變數（將內插到A grid)
        或是wrfout自身的變數(原始C grid位置)
        讀取wrfout網格資訊
        '''
        with Dataset(self.wrfout_fname) as filenc:
            self.dx = filenc.DX
            self.dy = filenc.DY
            self.Nx = filenc.dimensions['west_east'].size
            self.Ny = filenc.dimensions['south_north'].size
            self.center_xloc = ((self.Nx+1)/2 - 1) * self.dx
            self.center_yloc = ((self.Ny+1)/2 - 1) * self.dy

            for var in varlist:
                try:
                    exec(
                        f'self.{var} = np.squeeze(np.array(wrf.getvar(filenc,"{var}")))')
                    if var == 'P_HYD':
                        self.pressure = self.P_HYD
                        self.sfc_pressure = self.pressure[0]
                        #exec(f'self.{var} = wrf.smooth2d(self.{var},2)')

                except ValueError:
                    exec(
                        f'self.{var} = np.squeeze(np.array(filenc.variables["{var}"]))')

                except:
                    raise ValueError(
                        f"Variable name '{var}' is not available in this nc file.")


# 此為圓柱座標的網格系統
class cylindrical_grid(gridsystem):
    def init_grid(self):
        self.meta_convert_to_float('dr', 'dtheta', 'dz', 'dt',
                                   'z_start', 'r_start', 'theta_start')
        self.meta_convert_to_int('Nr', 'Ntheta', 'Nz')

        # 依照設定檔建立r、theta、z座標
        self.create_coord()

    def create_coord(self):
        self.r = np.arange(self.r_start, self.Nr*self.dr, self.dr)
        self.theta = np.arange(
            self.theta_start, self.Ntheta*self.dtheta, self.dtheta)
        self.z = np.arange(0, self.Nz*self.dz, self.dz) + self.z_start

        self.set_coord_meta(
            'z', 'vertical', self.z, unit='m',
            description='Vertical coordinate')
        self.set_coord_meta(
            'theta', 'tangential', self.theta, unit='rad',
            description='Tangential coordinate')
        self.set_coord_meta('r', 'radial', self.r, unit='m',
                            description='Radial coordinate')

    def set_horizontal_location(self, offset_x, offset_y):
        '''
        設定內插時的目標位置
        來源資料可能是直角坐標或是wrf座標，設定圓柱座標的水平格點在來源座標的x, y位置，供內插使用
        offset_x, offset_y是來源資料座標系中颱風中心的位置
        '''
        self.xTR, self.yTR = Make_cyclinder_coord(
            [offset_y, offset_x], self.r, self.theta)

    def set_terrain(self, var):
        # 決定地形高度，包含地形mask (self.isin_hgt)、地形idx (self.hgt_idx)
        self.hgt_idx = np.sum(np.isnan(var), axis=0)  # 決定地形idx，統計垂直方向的np.nan加總
        if self.z[0] == 0:  # 但有時候會在零星幾格單獨存在np.nan 要將這些慮除
            # 如果z座標底層高度是0，都會是np.nan，無法識別，所以識別z為第一層，如果這層存在np.nan，表示這裡才有地形，否則給予1 (為第0層的np.nan計數)
            self.hgt_idx = np.where(np.isnan(var[1, :, :]), self.hgt_idx, 1)
        else:
            # 如果z座標底層高度不是0，此層存在np.nan表示有地形，保留hgt_idx計數，否則將hgt_idx設為0 (np.nan為零星，非地形)
            self.hgt_idx = np.where(np.isnan(var[0, :, :]), self.hgt_idx, 0)
        self.isin_hgt = np.where(np.ones([self.Nz, self.Ntheta, self.Nr])*np.arange(
            self.Nz).reshape(-1, 1, 1) < self.hgt_idx, True, False)

    def set_data_old(self, wrf_dx, wrf_dy, wrf_vertical, **vardict):
        self.varlist3D = []
        self.varlist2D = []
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                tmp = np.array(wrf.interplevel(var, wrf_vertical, self.z))
                tmp2 = interp2D_fast_layers(
                    wrf_dx, wrf_dy, self.xTR, self.yTR, tmp)
                if self.varlist3D == []:
                    if self.vertical_coord_type == 'z':
                        self.set_terrain(tmp2)
                # tmp2 = self.fill_value(tmp2)
                exec(f'self.{varname} = tmp2')
                self.varlist3D.append(varname)
            elif len(var.shape) == 2:
                exec(
                    f'self.{varname} = interp2D_fast(wrf_dx, wrf_dy, self.xTR, self.yTR, var)')
                self.varlist2D.append(varname)

    def spline(self, var, vert, obj_vert):
        Nx = var.shape[2]
        Ny = var.shape[1]
        out = np.zeros([len(obj_vert), Ny, Nx])

        for j in range(Ny):
            for i in range(Nx):
                cs = splrep(vert[:, j, i], var[:, j, i], k=2)
                out[:, j, i] = splev(obj_vert, cs)

        return out

    @timer
    def set_data(
            self, wrf_dx, wrf_dy, wrf_vertical, ver_interp_order=1, **vardict):
        self.varlist3D = []
        self.varlist2D = []
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                self.varlist3D.append(varname)
            elif len(var.shape) == 2:
                exec(
                    f'self.{varname} = interp2D_fast(wrf_dx, wrf_dy, self.xTR, self.yTR, var)')
                self.varlist2D.append(varname)

        group = [[vardict[varname], wrf_vertical, self.z]
                 for varname in self.varlist3D]
        if ver_interp_order == 1:
            pool = mp.Pool(interp_cpus)
            tmp = pool.starmap(wrf.interplevel, group)
            pool.close()
            pool.join()
        elif ver_interp_order == 2:
            pool = mp.Pool(interp_cpus)
            tmp = pool.starmap(self.spline, group)
            pool.close()
            pool.join()
        for idx, varname in enumerate(self.varlist3D):
            #tmp = np.array(wrf.interplevel(var,wrf_vertical,self.z))
            tmp2 = interp2D_fast_layers(
                wrf_dx, wrf_dy, self.xTR, self.yTR, tmp[idx])
            # tmp2 = self.fill_value(tmp2)
            exec(f'self.{varname} = tmp2')

        if 'hgt' in self.varlist2D:
            self.set_terrain()  # 設定地形高度
            self.terrain_mask_vars()  # 將地形內部的格點設定為nan

    def set_terrain(self):
        def fix_hgt_idx(hgt_idx):
            hgt_idx[0, 1: -1] = np.where(hgt_idx[0, 1: -1] <
                                         hgt_idx[1, 1: -1],
                                         hgt_idx[1, 1: -1],
                                         hgt_idx[0, 1: -1])
            hgt_idx[-1, 1: -1] = np.where(hgt_idx[-1, 1: -1] <
                                          hgt_idx[-2, 1: -1],
                                          hgt_idx[-2, 1: -1],
                                          hgt_idx[-1, 1: -1])
            hgt_idx[1: -1, 0] = np.where(hgt_idx[1: -1, 0] <
                                         hgt_idx[1: -1, 1],
                                         hgt_idx[1: -1, 1],
                                         hgt_idx[1: -1, 0])
            hgt_idx[1: -1, -1] = np.where(hgt_idx[1: -1, -1] <
                                          hgt_idx[1: -1, -2],
                                          hgt_idx[1: -1, -2],
                                          hgt_idx[1: -1, -1])
            hgt_idx[0, 0] = np.max(
                [hgt_idx[0, 0], hgt_idx[1, 0], hgt_idx[0, 1]])
            hgt_idx[0, -1] = np.max([hgt_idx[0, -1],
                                    hgt_idx[1, -1], hgt_idx[0, -2]])
            hgt_idx[-1, 0] = np.max([hgt_idx[-1, 0],
                                    hgt_idx[-2, 0], hgt_idx[-1, 1]])
            hgt_idx[-1, -1] = np.max([hgt_idx[-1, -1],
                                     hgt_idx[-2, -1], hgt_idx[-1, -2]])
            return hgt_idx

        # hgt_mask為各網格是否在地形內，以wrfout內的地形高度HGT作為地形高度標準
        # hgt_idx為各水平位置點上，有多少個點位於地形之下
        self.hgt_idx = (self.hgt-self.z_start)//self.dz + 1
        self.hgt_idx = self.hgt_idx.astype('int32')
        self.hgt_idx = np.where(self.hgt_idx < 0, 0, self.hgt_idx)
        self.hgt_idx = fix_hgt_idx(self.hgt_idx)
        self.hgt_mask = np.where(
            self.hgt_idx.reshape(1, self.Ntheta, self.Nr) > np.arange(
                0, self.Nz, 1).reshape(-1, 1, 1),
            1, 0)

    def terrain_mask_vars(self):
        for varname in self.varlist3D:
            exec(
                f'self.{varname} = np.where(self.hgt_mask,np.nan,self.{varname})')

    def set_vr_vt(self):
        theta = self.theta.reshape(1, -1, 1)
        self.vr = self.v*np.sin(theta) + self.u*np.cos(theta)
        self.vt = self.v*np.cos(theta) - self.u*np.sin(theta)

    def set_fric_vr_vt(self):
        theta = self.theta.reshape(1, -1, 1)
        self.fric_vr = self.fric_v*np.sin(theta) + self.fric_u*np.cos(theta)
        self.fric_vt = self.fric_v*np.cos(theta) - self.fric_u*np.sin(theta)

    def calc_kinetic_energy(self):
        self.kinetic_energy = \
            (self.u**2 + self.v**2 + self.w**2)/2


class cartesian_grid(gridsystem):
    def init_grid(self):
        self.meta_convert_to_float('dx', 'dy', 'dz', 'z_start',
                                   'SOR_parameter', 'iteration_tolerance')
        self.meta_convert_to_int('Nx', 'Ny', 'Nz', 'max_iteration_times')

        # 設定x, y, z座標軸，及中心位置。
        self.create_coord()
        self.varlist3D = []
        self.varlist2D = []

    def create_coord(self):
        self.x = np.arange(0., self.Nx*self.dx, self.dx)
        self.y = np.arange(0., self.Ny*self.dy, self.dy)
        self.z = np.arange(0, self.Nz*self.dz, self.dz) + self.z_start
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.center_xloc = (self.x[0] + self.x[-1])/2
        self.center_yloc = (self.y[0] + self.y[-1])/2
        self.dis_from_center = (
            self.xx-self.center_xloc)**2 + (self.yy-self.center_yloc)**2

    def set_horizontal_location(self, offset_x, offset_y):
        '''
        設定內插時的目標位置
        來源資料可能是直角坐標或是wrf座標，設定此座標系的中心要在來源座標系的哪裡
        (offset_x, offset_y)，產生出此直角坐標系在來源座標系內的x, y位置，供內插使用
        '''
        self.hinterp_target_xloc = self.x + offset_x - self.center_xloc
        self.hinterp_target_yloc = self.y + offset_y - self.center_yloc
        self.hinterp_target_xxloc, self.hinterp_target_yyloc = \
            np.meshgrid(self.hinterp_target_xloc, self.hinterp_target_yloc)

    def spline(self, var, vert, obj_vert, order=2):
        '''
        針對網格垂直方向使用高階spline內插（預設order=2)
        '''
        Nx = var.shape[2]
        Ny = var.shape[1]
        out = np.zeros([len(obj_vert), Ny, Nx])

        for j in range(Ny):
            for i in range(Nx):
                cs = splrep(vert[:, j, i], var[:, j, i], k=order)
                out[:, j, i] = splev(obj_vert, cs)

        return out

    @timer
    def set_data(self, src_dx, src_dy, src_vertical, **vardict):
        '''
        將來源資料(vardict)內插到此網格座標系後，結果存於此網格
        需給定來源網格的src_dx, src_dy(需為固定單一值)，垂直位置(src_vertical)
        首次內插的變數必須包含hgt地形高度
        '''

        varlist3D_curr = []

        # 先處理2D陣列，3D陣列之後平行化處理
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                self.varlist3D.append(varname)
                varlist3D_curr.append(varname)
            elif len(var.shape) == 2:
                exec(
                    f'self.{varname} = interp2D_fast(src_dx, src_dy, self.hinterp_target_xxloc, self.hinterp_target_yyloc, var)')
                self.varlist2D.append(varname)

        # 3D平行化垂直內插
        group = [[vardict[varname], src_vertical, self.z]
                 for varname in varlist3D_curr]
        pool = mp.Pool(interp_cpus)
        tmp = pool.starmap(self.spline, group)
        pool.close()
        pool.join()

        # 3D水平內插
        for idx, varname in enumerate(varlist3D_curr):
            tmp2 = interp2D_fast_layers(
                src_dx, src_dy, self.hinterp_target_xxloc, self.hinterp_target_yyloc, tmp[idx])
            exec(f'self.{varname} = tmp2')

        if 'hgt' in self.varlist2D:
            self.set_terrain()  # 設定地形高度
            self.terrain_mask_vars()  # 將地形內部的格點設定為nan

    def set_terrain(self):
        def fix_hgt_idx(hgt_idx):
            hgt_idx[0, 1: -1] = np.where(hgt_idx[0, 1: -1] <
                                         hgt_idx[1, 1: -1],
                                         hgt_idx[1, 1: -1],
                                         hgt_idx[0, 1: -1])
            hgt_idx[-1, 1: -1] = np.where(hgt_idx[-1, 1: -1] <
                                          hgt_idx[-2, 1: -1],
                                          hgt_idx[-2, 1: -1],
                                          hgt_idx[-1, 1: -1])
            hgt_idx[1: -1, 0] = np.where(hgt_idx[1: -1, 0] <
                                         hgt_idx[1: -1, 1],
                                         hgt_idx[1: -1, 1],
                                         hgt_idx[1: -1, 0])
            hgt_idx[1: -1, -1] = np.where(hgt_idx[1: -1, -1] <
                                          hgt_idx[1: -1, -2],
                                          hgt_idx[1: -1, -2],
                                          hgt_idx[1: -1, -1])
            hgt_idx[0, 0] = np.max(
                [hgt_idx[0, 0], hgt_idx[1, 0], hgt_idx[0, 1]])
            hgt_idx[0, -1] = np.max([hgt_idx[0, -1],
                                    hgt_idx[1, -1], hgt_idx[0, -2]])
            hgt_idx[-1, 0] = np.max([hgt_idx[-1, 0],
                                    hgt_idx[-2, 0], hgt_idx[-1, 1]])
            hgt_idx[-1, -1] = np.max([hgt_idx[-1, -1],
                                     hgt_idx[-2, -1], hgt_idx[-1, -2]])
            return hgt_idx

        # hgt_mask為各網格是否在地形內，以wrfout內的地形高度HGT作為地形高度標準
        # hgt_idx為各水平位置點上，有多少個點位於地形之下
        self.hgt_idx = (self.hgt-self.z_start)//self.dz + 1
        self.hgt_idx = self.hgt_idx.astype('int32')
        self.hgt_idx = np.where(self.hgt_idx < 0, 0, self.hgt_idx)
        self.hgt_idx = fix_hgt_idx(self.hgt_idx)
        self.hgt_mask = np.where(
            self.hgt_idx.reshape(1, self.Ny, self.Nx) > np.arange(
                0, self.Nz, 1).reshape(-1, 1, 1),
            1, 0)

    def terrain_mask_vars(self):
        for varname in self.varlist3D:
            exec(
                f'self.{varname} = np.where(self.hgt_mask,np.nan,self.{varname})')
