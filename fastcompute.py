import numpy as np
import time as tm


if __name__ == 'meteotools.fastcompute':
    from .exceptions import InputError, DimensionError
    from . import fastcomputecore as fc
else:
    from exceptions import InputError, DimensionError
    import fastcomputecore as fc


def check_loc(loc,d,lend,dimname):
    '''
    檢查特定維度要內插的位置是否有超過資料範圍，會直接回報錯誤
    
    Parameters
    ----------
    loc : array
        要內插的位置
    d : float
        等間距網格座標的網格間距
    lend : int
        等間距網格座標的網格數
    dimname : str
        維度名稱

    Raises
    ------
    ValueError
        當內插位置超過資料範圍時，出現此錯誤

    Returns
    -------
    None.

    '''
    loc_max = np.nanmax(loc)
    loc_min = np.nanmin(loc)
    
    if np.nanmax(loc) > d*(lend-1):
        raise ValueError(f"{dimname}位置最大值 {loc_max} 超過網格範圍 0 ~ {d*(lend-1)}")
    if np.nanmin(loc) < 0:
        raise ValueError(f"{dimname}位置最小值 {loc_min} 超過網格範圍 0 ~ {d*(lend-1)}")


def interp2D_fast(dx,dy,xLoc,yLoc,data):
    '''
    將等間距的2D直角坐標雙線性內插至任意位置
    data的兩個維度網格間距必須相等

    Parameters
    ----------
    dx : float
        data 第1維(X)的網格間距
    dy : float
        data 第0維(Y)的網格間距
    xLoc : array
        要內插的目標點第1維(X)位置
    yLoc : array
        要內插的目標點第0維(Y)位置
    data : 2D array
        2維的等間距直角網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後新位置的陣列，維度與xLoc、yLoc相同

    '''
    if type(xLoc) != 'numpy.ndarray':
        xLoc = np.array(xLoc)
    if type(yLoc) != 'numpy.ndarray':
        yLoc = np.array(yLoc)

    if len(xLoc.shape) != len(yLoc.shape):  # 當目標位置為一維(軸)時，用meshgrid建立位置
        raise DimensionError("xLoc與yLoc維度需相同")
        
    check_loc(xLoc,dx,data.shape[-1],'X')
    check_loc(yLoc,dy,data.shape[-2],'Y')
    xLoc2 = np.where(np.isnan(xLoc),1.7e308,xLoc)
    yLoc2 = np.where(np.isnan(yLoc),1.7e308,yLoc)

    if len(data.shape) == 2:
        output = fc.interp2d_fast(dx,dy,xLoc2.reshape(-1),yLoc2.reshape(-1),data).reshape(xLoc.shape)
        return np.where(output==1.7e308,np.nan,output)

    else:
        raise DimensionError("來源資料(data)維度需為2")

def interp3D_fast(dx,dy,dz,xLoc,yLoc,zLoc,data):
    '''
    將等間距的2D直角坐標雙線性內插至任意位置
    data的兩個維度網格間距必須相等

    Parameters
    ----------
    dx : float
        data 第1維(X)的網格間距
    dy : float
        data 第0維(Y)的網格間距
    xLoc : array
        要內插的目標點第1維(X)位置
    yLoc : array
        要內插的目標點第0維(Y)位置
    data : 2D array
        2維的等間距直角網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後新位置的陣列，維度與xLoc、yLoc相同

    '''
    if type(xLoc) != 'numpy.ndarray':
        xLoc = np.array(xLoc)
    if type(yLoc) != 'numpy.ndarray':
        yLoc = np.array(yLoc)
    if type(zLoc) != 'numpy.ndarray':
        zLoc = np.array(zLoc)

    if (xLoc.shape == yLoc.shape) and (xLoc.shape == zLoc.shape) == False:  
        raise DimensionError("xLoc與yLoc與zLoc大小需相同")
        
    check_loc(xLoc,dx,data.shape[-1],'X')
    check_loc(yLoc,dy,data.shape[-2],'Y')
    check_loc(zLoc,dz,data.shape[-2],'z')
    xLoc2 = np.where(np.isnan(xLoc),1.7e308,xLoc)
    yLoc2 = np.where(np.isnan(yLoc),1.7e308,yLoc)
    zLoc2 = np.where(np.isnan(zLoc),1.7e308,zLoc)

    if len(data.shape) == 3:
        output = fc.interp2d_fast(dx,dy,dz,xLoc2.reshape(-1),yLoc2.reshape(-1),zLoc2.reshape(-1),data).reshape(xLoc.shape)
        return np.where(output==1.7e308,np.nan,output)

    else:
        raise DimensionError("來源資料(data)維度需為3")


def interp2D_fast_layers(dx,dy,xLoc,yLoc,data):
    '''
    將等間距的2D直角坐標雙線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後兩個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(xLoc) = (g,h,i)
         dim(yLoc) = (g,h,i)
         dim(output) = (a,b,c,d,g,h,i)
    (e,f) 為內插的維度，內插至(g,h,i)的位置
    (a,b,c,d)如時間、高度，沿這些維度重複內插(e,f)資料至(g,h,i)位置

    Parameters
    ----------
    dx : float
        data 倒數第1維(X)的網格間距
    dy : float
        data 倒數第2維(Y)的網格間距
    xLoc : array
        要內插的目標點倒數第1維(X)位置
    yLoc : array
        要內插的目標點倒數第2維(Y)位置
    data : array
        等間距直角網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後的陣列
    '''
    if type(xLoc) != 'numpy.ndarray':
        xLoc = np.array(xLoc)
    if type(yLoc) != 'numpy.ndarray':
        yLoc = np.array(yLoc)

    
    if len(xLoc.shape) != len(yLoc.shape):  
        raise DimensionError("xLoc與yLoc維度需相同")
    
    #檢查欲內插的位置是否有超出資料範圍
    check_loc(xLoc,dx,data.shape[-1],'X')
    check_loc(yLoc,dy,data.shape[-2],'Y')
    xLoc2 = np.where(np.isnan(xLoc),1.7e308,xLoc)
    yLoc2 = np.where(np.isnan(yLoc),1.7e308,yLoc)
    
    #維度處理，包含輸出的output維度、內插的layer層數
    output_dim = data.shape[:-2] + xLoc.shape
    layers = int(np.cumprod(data.shape[:-2])[-1])
    output = np.zeros([layers]+list(xLoc.shape))
    data = data.reshape([layers]+list(data.shape[-2:]))
    output = fc.interp2d_fast_layers(dx,dy,xLoc2.reshape(-1),yLoc2.reshape(-1),data).reshape(output_dim)
    output = np.where(output==1.7e308,np.nan,output)
    
    return output.reshape(output_dim)

def interp3D_fast_layers(dx,dy,dz,xLoc,yLoc,zLoc,data):
    '''
    將等間距的2D直角坐標雙線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後兩個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(xLoc) = (g,h,i)
         dim(yLoc) = (g,h,i)
         dim(output) = (a,b,c,d,g,h,i)
    (e,f) 為內插的維度，內插至(g,h,i)的位置
    (a,b,c,d)如時間、高度，沿這些維度重複內插(e,f)資料至(g,h,i)位置

    Parameters
    ----------
    dx : float
        data 倒數第1維(X)的網格間距
    dy : float
        data 倒數第2維(Y)的網格間距
    xLoc : array
        要內插的目標點倒數第1維(X)位置
    yLoc : array
        要內插的目標點倒數第2維(Y)位置
    data : array
        等間距直角網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後的陣列
    '''
    if type(xLoc) != 'numpy.ndarray':
        xLoc = np.array(xLoc)
    if type(yLoc) != 'numpy.ndarray':
        yLoc = np.array(yLoc)
    if type(zLoc) != 'numpy.ndarray':
        zLoc = np.array(zLoc)

    
    if (xLoc.shape == yLoc.shape) and (xLoc.shape == zLoc.shape) == False:  
        raise DimensionError("xLoc與yLoc與zLoc大小需相同")
    
    #檢查欲內插的位置是否有超出資料範圍
    check_loc(xLoc,dx,data.shape[-1],'X')
    check_loc(yLoc,dy,data.shape[-2],'Y')
    check_loc(zLoc,dz,data.shape[-3],'Z')
    xLoc2 = np.where(np.isnan(xLoc),1.7e308,xLoc)
    yLoc2 = np.where(np.isnan(yLoc),1.7e308,yLoc)
    zLoc2 = np.where(np.isnan(zLoc),1.7e308,zLoc)
    
    #維度處理，包含輸出的output維度、內插的layer層數
    output_dim = data.shape[:-3] + xLoc.shape
    layers = int(np.cumprod(data.shape[:-3])[-1])
    output = np.zeros([layers]+list(xLoc.shape))
    data = data.reshape([layers]+list(data.shape[-3:]))
    output = fc.interp3d_fast_layers(dx,dy,dz,xLoc2.reshape(-1),yLoc2.reshape(-1),zLoc2.reshape(-1),data).reshape(output_dim)
    output = np.where(output==1.7e308,np.nan,output)
    
    return output.reshape(output_dim)


def interp1D_fast_layers(dx,xLoc,data):
    '''
    將等間距的1D直角坐標線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後一個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(xLoc) = (g,h)
         dim(output) = (a,b,c,d,e,g,h)
    (f) 為內插的維度，內插至(g,h)的位置
    (a,b,c,d,e)如時間、高度，沿這些維度重複內插(e)資料至(g,h)位置

    Parameters
    ----------
    dx : float
        data 倒數第1維(X)的網格間距
    xLoc : array
        要內插的目標點倒數第1維(X)位置
    data : array
        等間距直角網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後的陣列
    '''
    if type(xLoc) != 'numpy.ndarray':
        xLoc = np.array(xLoc)
        
    #檢查欲內插的位置是否有超出資料範圍
    check_loc(xLoc,dx,data.shape[-1],'X')
    xLoc2 = np.where(np.isnan(xLoc),1.7e308,xLoc)
    
    #維度處理，包含輸出的output維度、內插的layer層數
    output_dim = data.shape[:-1] + xLoc.shape
    layers = int(np.cumprod(data.shape[:-1])[-1])
    output = np.zeros([layers]+list(xLoc.shape))
    data = data.reshape([layers]+list(data.shape[-1:]))

    output = fc.interp1d_fast_layers(dx,xLoc2.reshape(-1),data).reshape(output_dim)

    return np.where(output==1.7e308,np.nan,output)


def Make_cyclinder_coord(centerLocation,r,theta):
    '''
    建立圓柱座標的水平格點在直角坐標上的位置

    Parameters
    ----------
    centerLocation : [y位置,x位置]
        圓柱座標中心在直角坐標上的y位置與x位置
    r : array
        徑方向座標點
    theta : array
        切方向座標點 (rad，數學角)

    Returns
    -------
    array
        圓柱座標的水平格點在直角座標上的位置 (theta,r)

    '''    

    Nt = theta.shape[0]     #判斷給定的座標軸位置，座標軸可為交錯位置，因此Nt、data_Nr可能會少1
    Nr = r.shape[0]
    #建立圓柱座標系 r為要取樣的位置點
    
    #建立需要取樣的同心圓環狀位置(第1維:切向角度，第二維:中心距離)
    xThetaRLocation = np.zeros([Nt,Nr])               #初始化水平交錯後要取樣的x位置 水平交錯就是 +1 /2 dr
    yThetaRLocation = np.zeros([Nt,Nr])               #初始化水平交錯後要取樣的y位置
    
    #建立要取樣的座標位置(水平交錯)
    rr, ttheta = np.meshgrid(r, theta)
    xThetaRLocation = rr*np.cos(ttheta) + centerLocation[1]
    yThetaRLocation = rr*np.sin(ttheta) + centerLocation[0]
    
    return xThetaRLocation, yThetaRLocation


def interp1D(x_input,x_output,data):
    '''
    將不等間距的1D坐標線性內插至任意位置，data須為一維。

    Parameters
    ----------
    x_input : array
        data 的原始格點位置
    x_output : array
        要內插的目標點位置
    data : array
        不等間距網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後的陣列
    '''
    if type(x_output) != 'numpy.ndarray':
        x_output = np.array(x_output)

    #檢查x_input陣列是否是單調遞增、單調遞減
    x_input_d = np.array(x_input[1:]) - np.array(x_input[:-1])
    if (np.all(x_input_d>0) or np.all(x_input_d<0)) == False:
        raise ValueError('x_input位置陣列須為單調遞增或單調遞減')
    
    #檢查x_output內插目標位置是否有超過資料範圍
    x_output_max = np.nanmax(x_output)
    x_input_max = np.nanmax(x_input)
    x_output_min = np.nanmin(x_output)
    x_input_min = np.nanmin(x_input)
    if x_output_max > x_input_max:
        raise ValueError(f'x_output 位置最大值 {x_output_max} 超過資料範圍 {x_input_min} ~ {x_input_max}')
    if x_output_min < x_input_min:
        raise ValueError(f'x_output 位置最小值 {x_output_min} 超過資料範圍 {x_input_min} ~ {x_input_max}')
    
    x_output2 = np.where(np.isnan(x_output),1.7e308,x_output)

    if len(data.shape) == 1:
        output = fc.interp1d(x_input.reshape(-1),x_output2.reshape(-1),data).reshape(x_output.shape)
        return np.where(output==1.7e308,np.nan,output)
    else:
        raise DimensionError("來源資料(data)維度需為1")


def interp1D_layers(x_input,x_output,data):
    '''
    將不等間距的1D坐標線性內插至任意位置，並延其他維度都做相同的內插，如高度
    data的最後一個維度網格間距必須相等。
    例如: dim(data) = (a,b,c,d,e,f)
         dim(x_output) = (g,h)
         dim(output) = (a,b,c,d,e,g,h)
    (f) 為內插的維度，內插至(g,h)的位置
    (a,b,c,d,e)如時間、高度，沿這些維度重複內插(e)資料至(g,h)位置

    Parameters
    ----------
    x_input : array
        data 倒數第1維(X)的原始格點位置
    x_output : array
        要內插的目標點倒數第1維(X)位置
    data : array
        不等間距(最後一維)直角網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後的陣列
    '''
    if type(x_output) != 'numpy.ndarray':
        x_output = np.array(x_output)

    #檢查x_input陣列是否是單調遞增、單調遞減
    x_input_d = np.array(x_input[1:]) - np.array(x_input[:-1])
    if (np.all(x_input_d>0) or np.all(x_input_d<0)) == False:
        raise ValueError('x_input位置陣列須為單調遞增或單調遞減')
    
    #檢查x_output內插目標位置是否有超過資料範圍
    x_output_max = np.nanmax(x_output)
    x_input_max = np.nanmax(x_input)
    x_output_min = np.nanmin(x_output)
    x_input_min = np.nanmin(x_input)
    if x_output_max > x_input_max:
        raise ValueError(f'x_output 位置最大值 {x_output_max} 超過資料範圍 {x_input_min} ~ {x_input_max}')
    if x_output_min < x_input_min:
        raise ValueError(f'x_output 位置最小值 {x_output_min} 超過資料範圍 {x_input_min} ~ {x_input_max}')
    
    x_output2 = np.where(np.isnan(x_output),1.7e308,x_output)
    
    output_dim = data.shape[:-1] + x_output.shape
    layers = int(np.cumprod(data.shape[:-1])[-1])
    output = np.zeros([layers]+list(x_output.shape))
    data = data.reshape([layers]+list(data.shape[-1:]))
    for i in range(layers):
        output[i] = fc.interp1d(x_input.reshape(-1),x_output2.reshape(-1),data[i]).reshape(x_output.shape)
    output = np.where(output==1.7e308,np.nan,output)
    
    return output.reshape(output_dim)


def cartesian2cylindrical(dx,dy,car_data,centerLocation=[],r=[],theta=[],xTR=[],yTR=[]):
    '''
    將直角坐標內插到圓柱座標，只進行水平上的內插，並延續直角座標的垂直方向
    直角座標水平方向網格間距必須相等。
    
    Parameters
    ----------
    dx : float
        data 第2維(X)的網格間距
    dy : float
        data 第1維(Y)的網格間距
    car_data : 3D array
        直角座標資料
    centerLocation : [y位置,x位置], optional
         圓柱座標中心在直角坐標上的y位置與x位置 The default is [].
    r : array, optional
        徑方向座標點 The default is [].
    theta : array, optional
        切方向座標點 (rad，數學角) The default is [].
    xTR : 2D array, optional
        圓柱座標水平X位置 (theta,r) The default is [].
    yTR : 2D array, optional
        圓柱座標水平Y位置 (theta,r) The default is [].

    Raises
    ------
    InputError
        不足的座標資訊
    DimensionError
        資料維度錯誤
    Returns
    -------
    cyl_data : array
        內插後的圓柱座標資料 (z, theta, r)

    '''
    if len(car_data.shape) != 3:
        raise DimensionError('直角座標資料(car_data)須為3維')
        
    if xTR!=[] and yTR!=[]:
        lenr = xTR.shape[1]
        lenth = xTR.shape[0]
        lenz = car_data.shape[0]
    elif centerLocation!=[] and r!=[] and theta!=[]:
        lenr = r.shape[0]
        lenth = theta.shape[0]
        lenz = car_data.shape[0]
        xTR, yTR = Make_cyclinder_coord(centerLocation,r,theta)
    else:
        raise InputError("(xTR,yTR) 或 (centerLocation,r,theta) 須至少輸入一者")
        
    cyl_data = np.zeros([lenz,lenth,lenr])
    for k in range(lenz):
        cyl_data[k,:,:] = fc.interp2d_test(dx,dy,xTR.reshape(-1),yTR.reshape(-1),car_data[k,:,:]).reshape(xTR.shape)
    return cyl_data
    

def interp1D_fast(dx,xLoc,data):
    '''
    將等間距的1D資料雙線性內插至任意位置
    data的兩個維度網格間距必須相等

    Parameters
    ----------
    dx : float
        data 資料的網格間距
    xLoc : array
        要內插的目標點在原始資料的位置
    data : 1D array
        1維的等間距網格資料

    Raises
    ------
    DimensionError
        資料維度錯誤

    Returns
    -------
    array
        內插後新位置的陣列，維度與xLoc相同

    '''
    if type(xLoc) != 'numpy.ndarray':
        xLoc = np.array(xLoc)
        
    check_loc(xLoc,dx,data.shape[-1],'X')
    xLoc2 = np.where(np.isnan(xLoc),1.7e308,xLoc)
    if len(data.shape) == 1:
        output = fc.interp1d_fast(dx,xLoc2.reshape(-1),data).reshape(xLoc.shape)
        return np.where(output==1.7e308,np.nan,output)
    else:
        raise DimensionError("來源資料(data)維度需為1")


def Array_isthesame(a,b,tor=0):
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
        TF = (np.abs(a-b)<tor)
    TF = np.where(np.logical_and(np.isnan(a),np.isnan(b)), True, TF)
    if tor != 0 and a.dtype.name not in ['int32', 'int64', 'float32','float64']:
        print('Warning: Because two input arrays are not integer or float, tor is useless. Output True means all of the corresponding elements are the same.')
    if False in TF:
        return False
    else:
        return True


if __name__ == '__main__':
    '''
    a = np.zeros([3,7,5,3,8,4,5])
    b = np.zeros([4,3,2])
    c = np.zeros([4,3,2])
    for i in range(1,4):
        for j in range(1,8):
            for k in range(1,6):
                for l in range(1,4):
                    for m in range(1,9):
                        for n in range(1,5):
                            for o in range(1,6):
                                a[i-1,j-1,k-1,l-1,m-1,n-1,o-1] = o+10*n+100*m+1000*l+10000*k+100000*j+1000000*i
    b = np.random.rand(4,3,2)
    c = np.random.rand(4,3,2)
    c[2,1,1] = np.nan
    #d = interp1D_fast_layers(1,b,a)
    #e = interp1D_fast_layers(1,np.nan,a)
    
    interp2D_fast(1,1,c.reshape(-1),b.reshape(-1),a[0,0,0,0,0,:,:]).reshape(4,3,2)
    interp2D_fast(1,1,c.reshape(-1),b.reshape(-1),a[0,0,0,0,0,:,:]).reshape(4,3,2)
    #interp2D_fast(1,1,np.array([1]),np.array([1]),np.array([[1,1,1,],[1,1,1,],[1,1,1,]]))
    oo = interp2D_fast_layers(1,1,c,b,a)
    print(oo[1,1,1,1,1,1,1,1])
    #interp2D_fast_layers(1,1,b,c,a)
    '''
    '''
    a = np.array([0,1,2,3,4])
    b = np.array([1.5,2,3,4])
    c = np.array([[2,3,4,5,6],[2,3,4,5,6]])
    d = interp1D_fast(1,np.array([2,np.nan]),np.array([2,3,4,5,np.nan]))
    e = interp1D_fast_layers(1,b,c)
    '''
    '''
    a = np.zeros([10,10])
    for i in range(10):
        for j in range(10):
            a[i,j] = f'{i:d}{j:d}'
            
    y = [1,5,6]
    x = [5,3,8]
    '''
    import time as tm
    xLoc = [2.]
    yLoc = [2.]
    for i in range(5,501,5):
        data = np.zeros([97,810,810])
        data[5,100:500,:] = np.nan
        t0 = tm.time()
        a = interp2D_fast_layers(1.,1.,xLoc,yLoc,data)
        t1 = tm.time()
        print(i, i**3, t0-t1,np.sum(np.isnan(data)))
    
    '''
    import netCDF4 as nc
    import wrf
    import matplotlib.pyplot as plt
    
    
    
    filenc = nc.Dataset(r'D:\Research_data\WRF_data\LEKIMA4_new_WDM6_5min\wrfout_d04_2019-08-08_06_00_00')
    hdy = np.array(wrf.getvar(filenc,'H_DIABATIC'))
    dx=1000
    dy=1000
    lenx=810
    leny=810
    x = np.array(np.arange(0,lenx*dx,dx))
    y = np.array(np.arange(0,leny*dy,dy))
    centerLocation = [(y[0]+y[-1])/2,(x[0]+x[-1])/2]
    #設定座標系
    dr = 1000
    rmax = 400000
    
    dtheta = np.pi*2/360
    
    #生成座標軸
    r = np.arange(0,rmax+0.0001,dr)
    height = np.arange(500,20000.1,500)
    theta = np.arange(0,np.pi*2,dtheta)
    xTR2, yTR2 = Make_cyclinder_coord(centerLocation,r,theta)
    '''
    
    
