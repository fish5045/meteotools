import numpy as np
import multiprocessing as mp
import wrf
from scipy.interpolate import splev, splrep

from ..interpolation import interp2D_fast_layers, interp2D_fast
from ..calc import Make_cyclinder_coord, FD2
from ..tools import timer
from .grid_general import gridsystem
from .thermaldynamic_calculation import calc_thermaldynamic


class cylindrical_grid(gridsystem, calc_thermaldynamic):
    '''
    圓柱座標網格，3維網格(z, theta, r)，z可為壓力或高度座標，theta為切方向，r為徑方向。
    '''

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
            tmp = []
            for var, vertical, specific_z in group:
                tmp.append(np.array(wrf.interplevel(
                    var, vertical, specific_z)))

            '''
            pool = mp.Pool(self.interp_cpus)
            tmp = pool.map(wrf.interplevel,group)
            pool.close()
            pool.join()
            '''

        elif ver_interp_order == 2:
            pool = mp.Pool(self.interp_cpus)
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

    # ----------------------------------------------------
    # computation for cylinerical coordinates
    def calc_vr_vt(self):
        if 'u' in dir(self) and 'v' in dir(self):
            theta = self.theta.reshape(1, -1, 1)
            self.vr = self.v*np.sin(theta) + self.u*np.cos(theta)
            self.vt = self.v*np.cos(theta) - self.u*np.sin(theta)
        else:
            raise AttributeError('須先設定 u, v')

    def calc_fric_vr_vt(self):
        if 'fric_u' in dir(self) and 'fric_v' in dir(self):
            theta = self.theta.reshape(1, -1, 1)
            self.fric_vr = self.fric_v*np.sin(theta) + self.fric_u*np.cos(theta)
            self.fric_vt = self.fric_v*np.cos(theta) - self.fric_u*np.sin(theta)
        else:
            raise AttributeError('須先設定 fric_u, fric_v')

    def calc_kinetic_energy(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()
        if 'w' in dir(self):
            self.kinetic_energy = \
                (self.vr**2 + self.vt**2 + self.w**2)/2
        else:
            raise AttributeError('須先設定 w')

    def calc_vorticity_3D(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()
        if 'w' in dir(self):
            self.zeta_r = FD2(self.w, self.dtheta, 1)/self.r \
                - FD2(self.vt, self.dz, 0)
            self.zeta_t = FD2(self.vr, self.dz, 0) \
                - FD2(self.w, self.dr, 2)
            self.zeta_z = FD2(self.vt*self.r, self.dr, 2)/self.r \
                - FD2(self.vr, self.dtheta, 1)/self.r
        else:
            raise AttributeError('須先設定 w')

    def calc_vorticity_z(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()
        self.zeta_z = FD2(self.vt*self.r, self.dr, 2)/self.r \
            - FD2(self.vr, self.dtheta, 1)/self.r

    def calc_abs_vorticity_3D(self):
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_3D()
        if 'f' in dir(self):
            self.abs_zeta_z = self.zeta_z + self.f
        else:
            raise AttributeError('須先設定 f')

    def calc_abs_vorticity_z(self):
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_z()
        if 'f' in dir(self):
            self.abs_zeta_z = self.zeta_z + self.f
        else:
            raise AttributeError('須先設定 f')

    def calc_density_factor(self):
        if 'qv' in dir(self):
            a = 1 + self.qv/0.622
        else:
            raise AttributeError('須先設定 qv')

        if all(var in dir(self) for var in ['qc', 'qr', 'qi', 'qs', 'qg', 'qh']):
            b = 1 + self.qv + self.qc + self.qr + self.qi + self.qs + self.qg + self.qh
        elif all(var in dir(self) for var in ['qc', 'qr', 'qi', 'qs', 'qg']):
            b = 1 + self.qv + self.qc + self.qr + self.qi + self.qs + self.qg
        elif all(var in dir(self) for var in ['qc', 'qr', 'qi', 'qs']):
            b = 1 + self.qv + self.qc + self.qr + self.qi + self.qs
        elif all(var in dir(self) for var in ['qc', 'qr', 'qi']):
            b = 1 + self.qv + self.qc + self.qr + self.qi
        elif all(var in dir(self) for var in ['qc', 'qr']):
            b = 1 + self.qv + self.qc + self.qr
        elif all(var in dir(self) for var in ['qc']):
            b = 1 + self.qv + self.qc
        elif all(var in dir(self) for var in ['qr']):
            b = 1 + self.qv + self.qr
        elif all(var in dir(self) for var in []):
            b = 1
            print('Warning: 未設定液態固態水物混和比，計算出的Th_rho將與Th_v相同')
        self.density_factor = a/b

    def calc_density_potential_temperature(self):
        self.calc_density_factor()
        if 'Th' not in dir(self):
            self.calc_theta()   # 須繼承.thermaldynamic_calculation.calc_thermaldynamic
        self.Th_rho = self.Th * self.density_factor


    def calc_PV(self):
        if 'Th_rho' not in dir(self):
            self.calc_density_potential_temperature()
        if 'abs_zeta_z' not in dir(self):
            self.calc_abs_vorticity_3D()

        r_part = self.zeta_r * FD2(self.Th_rho, self.dr, 2)
        t_part = self.zeta_t * FD2(self.Th_rho,
                                    self.dtheta, 1) / self.r
        z_part = self.abs_zeta_z * FD2(self.Th_rho, self.dz, 0)
        self.PV = (r_part + t_part + z_part) / self.rho

# ------------------------------------
