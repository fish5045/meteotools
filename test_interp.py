import numpy as np
from meteotools.interpolation import interp1D_fast, interp1D_fast_layers
from meteotools.interpolation import interp1D_nonequal, interp1D_nonequal_layers
from meteotools.interpolation import interp2D_fast, interp2D_fast_layers
from meteotools.interpolation import interp3D_fast, interp3D_fast_layers
import os
import time


def interp1D(dx, xNewLoc, xData):
    original_xNewLoc_shape = xNewLoc.shape
    xNewLoc = xNewLoc.reshape(-1)
    RThetaData = np.zeros(xNewLoc.shape)
    for i in range(xNewLoc.shape[0]):
        x0 = int(xNewLoc[i]/dx)
        if ((xNewLoc[i]/dx) == int(xNewLoc[i]/dx)):
            x1 = x0
        else:
            x1 = x0 + 1

        Z0 = xData[x0]
        Z1 = xData[x1]
        RThetaData[i] = (Z1-Z0)*(xNewLoc[i]/dx-(x0-1))+Z0
    
    return RThetaData.reshape(original_xNewLoc_shape)

def interp1D_layers(dx, xNewLoc, xData):
    original_xNewLoc_shape = xNewLoc.shape
    xNewLoc = xNewLoc.reshape(-1)
    RThetaData = np.zeros([xData.shape[0], xNewLoc.shape[0]])
    for k in range(xData.shape[0]):
        for i in range(xNewLoc.shape[0]):
            x0 = int(xNewLoc[i]/dx)
            if ((xNewLoc[i]/dx) == int(xNewLoc[i]/dx)):
                x1 = x0
            else:
                x1 = x0 + 1
    
            Z0 = xData[k,x0]
            Z1 = xData[k,x1]
            RThetaData[k,i] = (Z1-Z0)*(xNewLoc[i]/dx-(x0-1))+Z0
    
    return RThetaData.reshape([xData.shape[0]] + list(original_xNewLoc_shape))

def interp2D(dx, dy, xNewLoc, yNewLoc, xyData):
    original_xNewLoc_shape = xNewLoc.shape
    xNewLoc = xNewLoc.reshape(-1)
    yNewLoc = yNewLoc.reshape(-1)
    RThetaData = np.zeros(xNewLoc.shape)
    for i in range(xNewLoc.shape[0]):
        x0 = int(xNewLoc[i]/dx)
        if ((xNewLoc[i]/dx) == int(xNewLoc[i]/dx)):
            x1 = x0
        else:
            x1 = x0 + 1
        y0 = int(yNewLoc[i]/dy)
        if ((yNewLoc[i]/dy) == int(yNewLoc[i]/dy)):
            y1 = y0
        else:
            y1 = y0 + 1

        Z00 = xyData[y0,x0]
        Z10 = xyData[y0,x1]
        Z01 = xyData[y1,x0]
        Z11 = xyData[y1,x1]
        tmpZ0 = (Z10-Z00)*(xNewLoc[i]/dx-(x0))+Z00
        tmpZ1 = (Z11-Z01)*(xNewLoc[i]/dx-(x0))+Z01
        RThetaData[i] = (tmpZ1-tmpZ0)*(yNewLoc[i]/dy-(y0))+tmpZ0
    
    return RThetaData.reshape(original_xNewLoc_shape)

def interp2D_layers(dx, dy, xNewLoc, yNewLoc, xyData):
    original_xNewLoc_shape = xNewLoc.shape
    xNewLoc = xNewLoc.reshape(-1)
    yNewLoc = yNewLoc.reshape(-1)
    RThetaData = np.zeros([xyData.shape[0], xNewLoc.shape[0]])
    for k in range(xyData.shape[0]):
        for i in range(xNewLoc.shape[0]):
            x0 = int(xNewLoc[i]/dx)
            if ((xNewLoc[i]/dx) == int(xNewLoc[i]/dx)):
                x1 = x0
            else:
                x1 = x0 + 1
            y0 = int(yNewLoc[i]/dy)
            if ((yNewLoc[i]/dy) == int(yNewLoc[i]/dy)):
                y1 = y0
            else:
                y1 = y0 + 1
    
            Z00 = xyData[k,y0,x0]
            Z10 = xyData[k,y0,x1]
            Z01 = xyData[k,y1,x0]
            Z11 = xyData[k,y1,x1]
            tmpZ0 = (Z10-Z00)*(xNewLoc[i]/dx-(x0))+Z00
            tmpZ1 = (Z11-Z01)*(xNewLoc[i]/dx-(x0))+Z01
            RThetaData[k,i] = (tmpZ1-tmpZ0)*(yNewLoc[i]/dy-(y0))+tmpZ0
    
    return RThetaData.reshape([xyData.shape[0]] + list(original_xNewLoc_shape))

def interp3D(dx, dy, dz, xNewLoc, yNewLoc, zNewLoc, xyzData):
    original_xNewLoc_shape = xNewLoc.shape
    xNewLoc = xNewLoc.reshape(-1)
    yNewLoc = yNewLoc.reshape(-1)
    zNewLoc = zNewLoc.reshape(-1)
    RThetaData = np.zeros(xNewLoc.shape)
    for i in range(xNewLoc.shape[0]):
        x0 = int(xNewLoc[i]/dx)
        if ((xNewLoc[i]/dx) == int(xNewLoc[i]/dx)):
            x1 = x0
        else:
            x1 = x0 + 1
        y0 = int(yNewLoc[i]/dy)
        if ((yNewLoc[i]/dy) == int(yNewLoc[i]/dy)):
            y1 = y0
        else:
            y1 = y0 + 1
        z0 = int(zNewLoc[i]/dz)
        if ((zNewLoc[i]/dz) == int(zNewLoc[i]/dz)):
            z1 = z0
        else:
            z1 = z0 + 1


        w000 = xyzData[z0,y0,x0]
        w010 = xyzData[z0,y1,x0]
        w001 = xyzData[z0,y0,x1]
        w011 = xyzData[z0,y1,x1]
        w100 = xyzData[z1,y0,x0]
        w110 = xyzData[z1,y1,x0]
        w101 = xyzData[z1,y0,x1]
        w111 = xyzData[z1,y1,x1]
        zfrac = zNewLoc[i]/dz - (z0)
        yfrac = yNewLoc[i]/dy - (y0)
        xfrac = xNewLoc[i]/dx - (x0)
        tmpw00 = (w100-w000)*zfrac + w000
        tmpw10 = (w110-w010)*zfrac + w010
        tmpw01 = (w101-w001)*zfrac + w001
        tmpw11 = (w111-w011)*zfrac + w011
        tmpw0 = (tmpw10-tmpw00)*yfrac + tmpw00
        tmpw1 = (tmpw11-tmpw01)*yfrac + tmpw01
        RThetaData[i] = (tmpw1-tmpw0)*xfrac + tmpw0
    
    return RThetaData.reshape(original_xNewLoc_shape)

def interp3D_layers(dx, dy, dz, xNewLoc, yNewLoc, zNewLoc, xyzData):
    original_xNewLoc_shape = xNewLoc.shape
    xNewLoc = xNewLoc.reshape(-1)
    yNewLoc = yNewLoc.reshape(-1)
    zNewLoc = zNewLoc.reshape(-1)
    RThetaData = np.zeros([xyzData.shape[0], xNewLoc.shape[0]])
    for k in range(xyzData.shape[0]):
        for i in range(xNewLoc.shape[0]):
            x0 = int(xNewLoc[i]/dx)
            if ((xNewLoc[i]/dx) == int(xNewLoc[i]/dx)):
                x1 = x0
            else:
                x1 = x0 + 1
            y0 = int(yNewLoc[i]/dy)
            if ((yNewLoc[i]/dy) == int(yNewLoc[i]/dy)):
                y1 = y0
            else:
                y1 = y0 + 1
            z0 = int(zNewLoc[i]/dz)
            if (zNewLoc[i] == int(zNewLoc[i]/dz)):
                z1 = z0
            else:
                z1 = z0 + 1
    
    
            w000 = xyzData[k,z0,y0,x0]
            w010 = xyzData[k,z0,y1,x0]
            w001 = xyzData[k,z0,y0,x1]
            w011 = xyzData[k,z0,y1,x1]
            w100 = xyzData[k,z1,y0,x0]
            w110 = xyzData[k,z1,y1,x0]
            w101 = xyzData[k,z1,y0,x1]
            w111 = xyzData[k,z1,y1,x1]
            zfrac = zNewLoc[i]/dz - (z0)
            yfrac = yNewLoc[i]/dy - (y0)
            xfrac = xNewLoc[i]/dx - (x0)
            tmpw00 = (w100-w000)*zfrac + w000
            tmpw10 = (w110-w010)*zfrac + w010
            tmpw01 = (w101-w001)*zfrac + w001
            tmpw11 = (w111-w011)*zfrac + w011
            tmpw0 = (tmpw10-tmpw00)*yfrac + tmpw00
            tmpw1 = (tmpw11-tmpw01)*yfrac + tmpw01
            RThetaData[k,i] = (tmpw1-tmpw0)*xfrac + tmpw0
    
    return RThetaData.reshape([xyzData.shape[0]] + list(original_xNewLoc_shape))

os.environ["OMP_NUM_THREADS"] = "8"  # 設置最大執行緒數
os.environ["OMP_DYNAMIC"] = "FALSE"  # 禁用動態執行緒分配
os.environ["OMP_SCHEDULE"] = "STATIC"  # 設定靜態調度（也可以試試 "DYNAMIC"）
os.environ["OMP_PROC_BIND"] = "false"
dx = 1.
dy = 3.
dz = 5.
N=100
x = np.arange(0,300,1.).astype(np.float32)
y = np.arange(0,90,1.).astype(np.float32)
z = np.arange(0,20,1.).astype(np.float32)
t = np.arange(0,10,1.).astype(np.float32)

xnew = np.random.rand(N,10).astype(np.float32)*299.
ynew = np.random.rand(N,10).astype(np.float32)*89.
znew = np.random.rand(N,10).astype(np.float32)*19.


# check interp1D
a = x.reshape(-1)
c = interp1D_fast(dx, xnew, a)
c2 = interp1D(dx,xnew,a)
assert(np.all((c-c2)/c < 1e-13))

# check interp1D_layers
a = x.reshape(1,-1) + y.reshape(-1,1)
c = interp1D_fast_layers(dx, xnew, a)
c2 = interp1D_layers(dx,xnew,a)
assert(np.all((c-c2)/c < 1e-13))


# check interp2D
a = x.reshape(1,-1) + y.reshape(-1,1)
c = interp2D_fast(dx, dy, xnew, ynew, a)
c2 = interp2D(dx,dy,xnew,ynew,a)
assert(np.all((c-c2)/c < 1e-13))

# check interp2D_layers
a = x.reshape(1,1,-1) + y.reshape(1,-1,1) + z.reshape(-1,1,1)
c = interp2D_layers(dx, dy, xnew, ynew, a)
c2 = interp2D_fast_layers(dx, dy, xnew, ynew, a)
assert(np.all((c-c2)/c < 1e-13))

# check interp3D
a = x.reshape(1,1,-1) + y.reshape(1,-1,1) + z.reshape(-1,1,1)
c = interp3D(dx, dy, dz, xnew, ynew, znew, a)
c2 = interp3D_fast(dx, dy, dz, xnew, ynew, znew, a.astype(np.float64))
assert(np.all((c-c2)/c < 1e-13))

# check interp3D
a = x.reshape(1,1,1,-1) + y.reshape(1,1,-1,1) \
    + z.reshape(1,-1,1,1)+ t.reshape(-1,1,1,1)
c = interp3D_layers(dx, dy, dz, xnew, ynew, znew, a)
c2 = interp3D_fast_layers(dx, dy, dz, xnew, ynew, znew, a)
assert(np.all((c-c2)/c < 1e-13))

# check interp1D nonequal
x = np.array([0,1,2,4,8,16,32])
a = x*2
xnew = np.random.rand(N,10).astype(np.float32)*32.
c = interp1D_nonequal(x,xnew,a)
assert(np.all((c-xnew*2)/c < 1e-13))

# check interp1D nonequal layers
x = np.arange(0,30000.,1) + np.random.rand(30000)*0.8
a = np.stack([x*2]*150)
xnew = np.random.rand(N,10).astype(np.float64)*29980. + 10
xnew[0,0] = x[0]
xnew[0,1] = x[-1]
time1 = time.time()
c = interp1D_nonequal_layers(x,xnew,a)
time2 = time.time()
print(time2-time1)
assert(np.all((c-xnew*2)/c < 1e-13))




