import numpy as np
from exceptions import UnitError, DimensionError
#from . import modelf as mf
import wrf


def Make_vertical_axis(heightstart, heightend, dp, inunit='m'):
    '''
    建立垂直座標軸(輸出標準單位，高度座標: m，壓力座標: Pa)
    輸入: 起始、終止、間距
    輸出: 非交錯垂直座標      height         [lenheight]
         交錯垂直座標        heights        [lenheight-1]
    選項: 輸入的單位         inunit           'm'
    '''

    #檢查單位、單位轉換
    if inunit not in ['m', 'Pa', 'km', 'hPa']:
        raise UnitError('不支援的單位')
    if inunit == 'km':
        heightstart *= 1000
        heightend *= 1000
        dp *= 1000
    elif inunit == 'hPa':
        heightstart *= 100
        heightend *= 100
        dp *= 100

    #建立垂直座標
    height = np.arange(heightstart, heightend+0.01, dp, dtype='float64')
    heights = height[:-1] + dp/2

    return height, heights


def Make_tangential_axis(thetastart, thetaend, dtheta, inunit='rad'):
    '''
    建立切向座標軸(圓柱座標; 輸出標準單位:數學角rad)
    輸入: 起始、終止、間距
    輸出: 非交錯切向座標        theta        [Nt]
         交錯切向座標          thetas       [Nt-1] or [Nt](完整一圈)
    選項: 輸入的單位           inunit       'rad'
    '''

    #檢查單位、單位轉換
    if inunit not in ['deg', 'rad']:
        raise UnitError('不支援的單位')
    if inunit == 'deg':
        thetastart *= np.pi/180
        thetaend *= np.pi/180
        dtheta *= np.pi/180

    #建立切向座標(圓柱座標)
    theta = np.arange(thetastart, thetaend+0.00001, dtheta, dtype='float64')
    thetas = (theta[:-1] + theta[1:])/2
    if thetaend-thetastart == 2*np.pi:
        theta = theta[:-1]

    return theta, thetas


def Make_radial_axis(rmin, rmax, dr, inunit='m'):
    '''
    建立徑向座標軸(圓柱座標; 輸出標準單位:m)
    輸入: 起始、終止、間距
    輸出: 非交錯徑向座標      r         [Nr]
         交錯徑向座標        rs        [Nr-1]
    選項: 輸入的單位         inunit    'm'
    '''

    #檢查單位、單位轉換
    if inunit not in ['m', 'km']:
        raise UnitError('不支援的單位')
    if inunit == 'km':
        rmin *= 1000
        rmax *= 1000
        dr *= 1000

    #建立徑向座標(圓柱座標)
    r = np.arange(rmin, rmax+0.0001, dr, dtype='float64')
    rs = r[:-1] + dr/2
    return r, rs


def Array_is_the_same(a, b, tor=0):
    '''
    判定兩陣列的所有元素是否完全一樣(維度、大小、數值，不包含nan)
    輸入: 陣列一
         陣列二
    輸出: bool
    選項: 浮點數容許的round off error 範圍 (tor)
    '''

    if a.shape != b.shape:
        return False
    if a.dtype.name != b.dtype.name:
        return False
    if tor == 0 or a.dtype.name not in ['int32', 'int64', 'float32','float64']:
        TF = (a == b)
    else:
        TF = (np.abs(a-b) < tor)

    #numpy 中即使對應元素皆為nan，仍為False，因此取代為true
    TF = np.where(np.logical_and(np.isnan(a), np.isnan(b)), True, TF)
    if tor != 0 and a.dtype.name not in ['int32', 'int64', 'float32', 'float64']:
        print('Warning: Because two input arrays are not integer or float, tor is useless. Output True means all of the corresponding elements are the same.')
    if False in TF:
        return False
    else:
        return True


def interp(dx, dy, xLoc, yLoc, data):
    '''
    將直角網格座標線性內插至任意位置，來源資料可為三維(或二維)，但目標位置為二維，將內插到該位置的不同高度層。
    輸入: 來源資料的x方向網格大小          dx
         來源資料的x方向網格大小          dy
         要內插的位置x座標 (二維以下)     xLoc      [dim0, dim1]
         要內插的位置y座標 (二維以下)     yLoc      [dim0, dim1]
         來源資料，直角網格              data      [Nz, Ny, Nx] 或 [Ny, Nx]
    輸出: 內插後資料                    output    [lenz,dim0,dim1] 或 [dim0,dim1]
    '''
    if len(xLoc.shape) != len(yLoc.shape):  # 當目標位置為一維(軸)時，用meshgrid建立位置
        raise DimensionError("xLoc與yLoc維度需相同")
    elif len(xLoc.shape) == 1 and len(yLoc.shape) == 1:
        xLoc, yLoc = np.meshgrid(xLoc, yLoc)

    if len(data.shape) == 2:
        size0 = data.shape[0]
        size1 = data.shape[1]
        output = mf.interp(dx, dy, xLoc, yLoc, data.reshape(1, size0, size1))
        return np.squeeze(output)

    elif len(data.shape) == 3:
        output = mf.interp(dx, dy, xLoc, yLoc, data)
        return output
    else:
        raise DimensionError("來源資料(data)維度需為2或3")


def Make_cyclinder_coord(centerLocation,r,theta):
    '''
    以颱風中心為原點，建立水平圓柱座標系的(theta,r)對應的直角座標位置
    輸入: centerLocation        颱風中心位置(y,x)      [1,1]
         r                     徑向座標               [lenr]
         theta                 徑向座標               [lentheta]
    '''
    #建立圓柱座標系 r為要取樣的位置點
    #建立需要取樣的同心圓環狀位置(第1維:切向角度，第二維:中心距離)
    #建立要取樣的座標位置(水平交錯)
    xThetaRLocation = r.reshape(1,-1)*np.cos(theta).reshape(-1,1) \
                      + centerLocation[1]
    yThetaRLocation = r.reshape(1,-1)*np.sin(theta).reshape(-1,1) \
                      + centerLocation[0]

    return xThetaRLocation, yThetaRLocation


def Get_wrf_data_cyclinder(filenc, varname, pre, interpHeight,
                           xinterpLoc=np.array([np.nan]), yinterpLoc=np.array([np.nan]),
                           interp2cylinder=True, wrfvar=[]):
    '''
    讀取WRF資料(並內插至圓柱座標,optional)
    輸入: WRF output nc檔案          filenc           nc_file
         要讀取的WRF變數(文字)        varname          str
         該WRF檔案的壓力座標          pre              wrf_var
         要內插到的壓力層             interpHeight     [lenheight]
    輸出: wrf變數                    wrfvar           wrf_var
         讀取的資料                  var...
    選項:
         要內插到的圓柱座標x位置       xinterpLoc       [Nt,Nr]    預設 np.array([np.nan])
         要內插到的圓柱座標y位置       yinterpLoc       [Nt,Nr]    預設 np.array([np.nan])
         是否使用內插                interp2cylinder             預設 True
         使用使用給定的wrf變數        wrfvar           wrf_var    預設 []
    注意: 若使用內插，xinterpLo與yinterpLoc需給定，此時回傳圓柱座標資料    [lenheight,Nt,Nr]
         不使用內插，回傳等壓直角座標                                   [lenheight,leny,lenx]
    '''
    if interp2cylinder==True and \
        (Array_is_the_same(xinterpLoc, np.array([np.nan]))
         or Array_is_the_same(xinterpLoc, np.array([np.nan]))):
        raise RuntimeError("Fatal Error: Interpolation Function is on but given not enough locations information (xinterpLoc, yinterpLoc).")

    if wrfvar == []:        # 讀取WRF變數
        try:
            wrfvar = np.array(wrf.getvar(filenc, varname))
        except IndexError:
            try:
                wrfvar = np.array(wrf.getvar(filenc, varname))
            except IndexError:
                wrfvar = np.array(wrf.getvar(filenc, varname))
                print("xf, xt, yf, yt are invaild and use the whole grids.")


    dim = len(wrfvar.shape) # 依照資料的不同維度，有不同功能

    if dim < 2:             # 資料小於2維，直接回傳
        return wrfvar

    elif dim == 2:          # 資料為2維 (單層)，可選擇內插到圓柱座標
        if interp2cylinder == False:
            return wrfvar, np.array(wrfvar)
        else:
            return wrfvar, interp(filenc.getncattr('DX'),
                                  filenc.getncattr('DY'),
                                  xinterpLoc,yinterpLoc,
                                  np.array(wrfvar))

    elif dim == 3:         # 資料為3維，可選擇內插到哪幾層高度，可選擇內插到圓柱座標
        var_pre = wrf.interplevel(wrfvar,pre,interpHeight,missing=np.nan)
        if interp2cylinder == False:
            return wrfvar, var_pre
        else:
            return wrfvar, interp(filenc.getncattr('DX'),
                                  filenc.getncattr('DY'),
                                  xinterpLoc,yinterpLoc,
                                  var_pre)


def Nine_pts_smooth(var):
    '''
    九點平滑網格資料
    輸入: 網格資料      var       二維陣列
    輸出: 網格資料      var       二維陣列

    '''
    svar = np.zeros([var.shape[0]+2, var.shape[1]+2])
    svar[1:-1,1:-1] = var
    svar[0,1:-1] = var[0,:]
    svar[-1,1:-1] = var[-1,:]
    svar[1:-1,0] = var[:,0]
    svar[1:-1,-1] = var[:,-1]
    svar[ 0, 0] = var[ 0, 0]
    svar[-1, 0] = var[-1, 0]
    svar[ 0,-1] = var[ 0,-1]
    svar[-1,-1] = var[-1,-1]

    var = (svar[:-2,:-2] + svar[:-2,1:-1] + svar[:-2,2:]
           + svar[1:-1,:-2] + svar[1:-1,1:-1] + svar[1:-1,2:]
           + svar[2:,:-2] + svar[2:,1:-1] + svar[2:,2:])/9.
    return var


def Three_pts_smooth_H(var):
    '''
    第二維方向(水平)三點平滑網格資料
    輸入: 網格資料      var       二維陣列
    輸出: 網格資料      var       二維陣列
    '''

    svar = np.zeros([var.shape[0], var.shape[1]+2])
    svar[:,1:-1] = var
    svar[:,0] = var[:,0]
    svar[:,-1] = var[:,-1]
    var = (svar[:,:-2] + svar[:,1:-1] + svar[:,2:])/3.
    return var

def Three_pts_smooth_V(var):
    '''
    第一維方向(垂直)三點平滑網格資料
    輸入: 網格資料      var       二維陣列
    輸出: 網格資料      var       二維陣列
    '''

    svar = np.zeros([var.shape[0]+2, var.shape[1]])
    svar[1:-1:,:] = var
    svar[0,:] = var[0,:]
    svar[-1,:] = var[-1,:]
    var = (svar[:-2,:] + svar[1:-1,:] + svar[2:,:])/3.
    return var



#if __name__ == '__main__':

