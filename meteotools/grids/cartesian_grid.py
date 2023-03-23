from scipy.interpolate import splev, splrep
import numpy as np
import multiprocessing as mp
from ..interpolation import interp2D_fast_layers, interp2D_fast


from .grid_general import gridsystem
from ..tools import timer


class cartesian_grid(gridsystem):
    '''
    圓柱座標網格，3維網格(z, y, x)，z可為壓力或高度座標，x, y為水平直角等間距座標。
    '''

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
        pool = mp.Pool(self.interp_cpus)
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
