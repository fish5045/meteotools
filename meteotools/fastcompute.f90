!gfortran -shared -fPIC -fopenmp -lgomp fastcompute.f90 -o lib/fastcompute.dll -O3 -pthread

module fastcompute
use iso_c_binding
use omp_lib

contains

    subroutine interp2D_fast(lenx,leny,dx,dy,N,xNewLoc,yNewLoc,xyData,RThetaData)  bind(c,name='interp2D_fast')

    implicit none
    real(c_double) :: Z00, Z01, Z10, Z11, tmpZ0, tmpZ1
    integer(c_int) :: x0, y0, x1, y1
    integer(c_int) :: k,j,i
    integer(c_int),intent(in), value :: lenx, leny, N   ! 分別表示直角座標x、y、z，圓柱座標theta, r
    real(c_double),intent(in), value :: dx,dy
    real(c_double),dimension(N), intent(in) :: xNewLoc, yNewLoc
    real(c_double),dimension(leny,lenx), intent(in) :: xyData
    real(c_double),dimension(N) ,intent(out) :: RThetaData
    !!$ integer(c_int), intent(out) :: num_threads

    !!$ num_threads = omp_get_max_threads()
    !!$call omp_set_num_threads(num_threads)
    !$omp parallel do private(i,x0,y0,x1,y1,Z00,Z10,Z01,Z11,tmpZ0,tmpZ1)
    do i = 1, N
        ! 處理xNewLoc為無效值的情況，直
        if ((xNewLoc(i) == 1.7D308) .or. (yNewLoc(i) == 1.7D308)) then
        RThetaData(i) = 1.7D308
        cycle
        end if
        !決定內插的index，若為最後一格，x1 = x0 = lenx
        x0 = int(xNewLoc(i)/dx)+1
        if ((xNewLoc(i)/dx) == int(xNewLoc(i)/dx)) then
            x1 = x0
        else
            x1 = x0 + 1
        end if
        y0 = int(yNewLoc(i)/dy)+1
        if ((yNewLoc(i)/dy) == int(yNewLoc(i)/dy)) then
            y1 = y0
        else
            y1 = y0 + 1
        end if

        !內插計算
        Z00 = xyData(y0,x0)
        Z10 = xyData(y0,x1)
        Z01 = xyData(y1,x0)
        Z11 = xyData(y1,x1)
        tmpZ0 = (Z10-Z00)*(xNewLoc(i)/dx-(x0-1))+Z00
        tmpZ1 = (Z11-Z01)*(xNewLoc(i)/dx-(x0-1))+Z01
        RThetaData(i) = (tmpZ1-tmpZ0)*(yNewLoc(i)/dy-(y0-1))+tmpZ0

    end do
    !$omp end parallel do

    return
    end subroutine interp2D_fast

    subroutine interp2D_fast_layers(lenx,leny,layers,dx,dy,N,xNewLoc,yNewLoc,xyData,RThetaData) bind(c,name='interp2D_fast_layers')
    use iso_c_binding
    implicit none
    real(c_double) :: Z00, Z01, Z10, Z11, tmpZ0, tmpZ1, xtmp, ytmp
    integer(c_int) :: x0, y0, x1, y1
    integer(c_int) :: k,i
    integer(c_int),intent(in), value :: lenx, leny, N, layers   ! 分別表示直角座標x、y、z，圓柱座標theta, r
    real(c_double),intent(in), value :: dx,dy
    real(c_double),intent(in) :: xNewLoc(N), yNewLoc(N), xyData(layers,leny,lenx)
    real(c_double),intent(out) :: RThetaData(layers,N)

    !$omp parallel do private(k,i,x0,y0,x1,y1,Z00,Z10,Z01,Z11,tmpZ0,tmpZ1) collapse(2) firstprivate(dx,dy)
    do k = 1, layers
        do i = 1, N
            ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
            if ((xNewLoc(i) == 1.7D308) .or. (yNewLoc(i) == 1.7D308)) then
            RThetaData(k,i) = 1.7D308
            cycle
            end if
            !決定內插的index，若為最後一格，x1 = x0 = lenx
            xtmp = xNewLoc(i)/dx
            ytmp = yNewLoc(i)/dy
            x0 = int(xtmp)+1
            if ((xtmp) == int(xtmp)) then
                x1 = x0
            else
                x1 = x0 + 1
            end if
            y0 = int(ytmp)+1
            if ((ytmp) == int(ytmp)) then
                y1 = y0
            else
                y1 = y0 + 1
            end if

            !內插計算
            Z00 = xyData(k,y0,x0)
            Z10 = xyData(k,y0,x1)
            Z01 = xyData(k,y1,x0)
            Z11 = xyData(k,y1,x1)
            tmpZ0 = (Z10-Z00)*(xNewLoc(i)/dx-(x0-1))+Z00
            tmpZ1 = (Z11-Z01)*(xNewLoc(i)/dx-(x0-1))+Z01
            RThetaData(k,i) = (tmpZ1-tmpZ0)*(yNewLoc(i)/dy-(y0-1))+tmpZ0

        end do
    end do
    !$omp end parallel do

    return
    end subroutine

    subroutine interp3D_fast(lenx,leny,lenz,dx,dy,dz,N,xNewLoc,yNewLoc,zNewLoc,xyzData,NewData) bind(c,name='interp3D_fast')
    implicit none
    real(c_double) :: w000, w001, w010, w011, w100, w101, w110, w111
    real(c_double) :: tmpw00, tmpw10, tmpw01, tmpw11, tmpw0, tmpw1
    real(c_double) :: xfrac, yfrac, zfrac
    integer(c_int) :: x0, y0, z0, x1, y1, z1
    integer(c_int) :: i
    integer(c_int),intent(in),value :: lenx, leny, lenz, N   ! 分別表示直角座標x、y、z，新的內插位置數目
    real(c_double),intent(in),value :: dx, dy, dz
    real(c_double),intent(in) :: xNewLoc(N),yNewLoc(N),zNewLoc(N), xyzData(lenz, leny, lenx)
    real(c_double),intent(out) :: NewData(N)

    !$omp parallel do private(i, w000, w001, w010, w011, w100, w101, w110, w111, &
    !$omp                     tmpw00, tmpw10, tmpw01, tmpw11, tmpw0, tmpw1, xfrac, yfrac, zfrac, &
    !$omp                     x0, y0, z0, x1, y1, z1)
    do i = 1, N
            ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
            if (((xNewLoc(i) == 1.7D308) .or. (yNewLoc(i) == 1.7D308)) .or. (zNewLoc(i) == 1.7D308)) then
                NewData(i) = 1.7D308
                cycle
            end if

            !決定內插的index，若為最後一格，x1 = x0 = lenx
            x0 = int(xNewLoc(i)/dx)+1
            if (xNewLoc(i) == (lenx-1)*dx) then
                x1 = x0
            else
                x1 = x0 + 1
            end if
            y0 = int(yNewLoc(i)/dy)+1
            if (yNewLoc(i) == (leny-1)*dy) then
                y1 = y0
            else
                y1 = y0 + 1
            end if
            z0 = int(zNewLoc(i)/dz)+1
            if (zNewLoc(i) == (lenz-1)*dz) then
                z1 = z0
            else
                z1 = z0 + 1
            end if

            !內插計算
            w000 = xyzData(z0,y0,x0)
            w010 = xyzData(z0,y1,x0)
            w001 = xyzData(z0,y0,x1)
            w011 = xyzData(z0,y1,x1)
            w100 = xyzData(z1,y0,x0)
            w110 = xyzData(z1,y1,x0)
            w101 = xyzData(z1,y0,x1)
            w111 = xyzData(z1,y1,x1)
            !內插Z方向
            zfrac = zNewLoc(i)/dz - (z0-1)
            yfrac = yNewLoc(i)/dy - (y0-1)
            xfrac = xNewLoc(i)/dx - (x0-1)
            tmpw00 = (w100-w000)*zfrac + w000
            tmpw10 = (w110-w010)*zfrac + w010
            tmpw01 = (w101-w001)*zfrac + w001
            tmpw11 = (w111-w011)*zfrac + w011
            tmpw0 = (tmpw10-tmpw00)*yfrac + tmpw00
            tmpw1 = (tmpw11-tmpw01)*yfrac + tmpw01
            NewData(i) = (tmpw1-tmpw0)*xfrac + tmpw0
    end do
    !$omp end parallel do

    return
    end subroutine

    subroutine interp3D_fast_layers(lenx,leny,lenz,layers,dx,dy,dz,N,xNewLoc,yNewLoc,zNewLoc,xyzData,NewData) &
        bind(c,name='interp3D_fast_layers')
    implicit none
    real(c_double) :: w000, w001, w010, w011, w100, w101, w110, w111
    real(c_double) :: tmpw00, tmpw10, tmpw01, tmpw11, tmpw0, tmpw1
    real(c_double) :: xfrac, yfrac, zfrac
    integer(c_int) :: x0, y0, z0, x1, y1, z1
    integer(c_int) :: k,i
    integer(c_int),intent(in),value :: lenx, leny, lenz, N, layers   ! 分別表示直角座標x、y、z，新的內插位置數目
    real(c_double),intent(in),value :: dx, dy, dz
    real(c_double),intent(in) :: xNewLoc(N),yNewLoc(N),zNewLoc(N), xyzData(layers, lenz, leny, lenx)
    real(c_double),intent(out) :: NewData(layers, N)

    !$omp parallel do private(k, i, w000, w001, w010, w011, w100, w101, w110, w111, &
    !$omp                     tmpw00, tmpw10, tmpw01, tmpw11, tmpw0, tmpw1, xfrac, yfrac, zfrac, &
    !$omp                     x0, y0, z0, x1, y1, z1)
    do k = 1, layers
        do i = 1, N
            ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
            if (((xNewLoc(i) == 1.7D308) .or. (yNewLoc(i) == 1.7D308)) .or. (zNewLoc(i) == 1.7D308)) then
                NewData(k,i) = 1.7D308
                cycle
            end if

            !決定內插的index，若為最後一格，x1 = x0 = lenx
            x0 = int(xNewLoc(i)/dx)+1
            if (xNewLoc(i) == (lenx-1)*dx) then
                x1 = x0
            else
                x1 = x0 + 1
            end if
            y0 = int(yNewLoc(i)/dy)+1
            if (yNewLoc(i) == (leny-1)*dy) then
                y1 = y0
            else
                y1 = y0 + 1
            end if
            z0 = int(zNewLoc(i)/dz)+1
            if (zNewLoc(i) == (lenz-1)*dz) then
                z1 = z0
            else
                z1 = z0 + 1
            end if

            !內插計算
            w000 = xyzData(k,z0,y0,x0)
            w010 = xyzData(k,z0,y1,x0)
            w001 = xyzData(k,z0,y0,x1)
            w011 = xyzData(k,z0,y1,x1)
            w100 = xyzData(k,z1,y0,x0)
            w110 = xyzData(k,z1,y1,x0)
            w101 = xyzData(k,z1,y0,x1)
            w111 = xyzData(k,z1,y1,x1)
            !內插Z方向
            zfrac = zNewLoc(i)/dz - (z0-1)
            yfrac = yNewLoc(i)/dy - (y0-1)
            xfrac = xNewLoc(i)/dx - (x0-1)
            tmpw00 = (w100-w000)*zfrac + w000
            tmpw10 = (w110-w010)*zfrac + w010
            tmpw01 = (w101-w001)*zfrac + w001
            tmpw11 = (w111-w011)*zfrac + w011
            tmpw0 = (tmpw10-tmpw00)*yfrac + tmpw00
            tmpw1 = (tmpw11-tmpw01)*yfrac + tmpw01
            NewData(k,i) = (tmpw1-tmpw0)*xfrac + tmpw0
        end do
    end do
    !$omp end parallel do

    return
    end subroutine


    subroutine interp1D_fast(lenx,dx,Nr,xNewLoc,xData,RThetaData) bind(c,name='interp1D_fast')
    implicit none
    real(c_double) :: Z0, Z1
    integer(c_int) :: x0, x1
    integer(c_int) :: i
    integer(c_int),intent(in), value :: lenx, Nr   ! 分別表示直角座標x、y、z，圓柱座標theta, r
    real(c_double),intent(in), value :: dx
    real(c_double),intent(in) :: xNewLoc(Nr), xData(lenx)
    real(c_double),intent(out) :: RThetaData(Nr)

    !$omp parallel do private(x0,x1,i,Z0,Z1)
    do i = 1, Nr
        ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
        if (xNewLoc(i) == 1.7D308) then
            RThetaData(i) = 1.7D308
            cycle
        end if
        !決定內插的index，若為最後一格，x1 = x0 = lenx
        x0 = int(xNewLoc(i)/dx)+1
        if (xNewLoc(i) == (lenx-1)*dx) then
            x1 = x0
        else
            x1 = x0 + 1
        end if

        !內插計算
        Z0 = xData(x0)
        Z1 = xData(x1)
        RThetaData(i) = (Z1-Z0)*(xNewLoc(i)/dx-(x0-1))+Z0

    end do
    !$omp end parallel do
    return
    end subroutine


    subroutine interp1D_fast_layers(lenx,layers,dx,Nr,xNewLoc,xData,NewData) bind(c,name='interp1D_fast_layers')
    implicit none
    real(c_double) :: Z0, Z1
    integer(c_int) :: x0, x1
    integer(c_int) :: i,k
    integer(c_int),intent(in), value :: lenx, Nr,layers   ! 分別表示直角座標x、y、z，圓柱座標theta, r
    real(c_double),intent(in), value :: dx
    real(c_double),intent(in) :: xNewLoc(Nr), xData(layers,lenx)
    real(c_double),intent(out) :: NewData(layers,Nr)


    !$omp parallel do private(k,x0,x1,i,Z0,Z1)
    do k = 1,layers
        do i = 1, Nr
            ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
            if (xNewLoc(i) == 1.7D308) then
                NewData(k,i) = 1.7D308
                cycle
            end if

            !決定內插的index，若為最後一格，x1 = x0 = lenx
            x0 = int(xNewLoc(i)/dx)+1
            if (xNewLoc(i) == (lenx-1)*dx) then
                x1 = x0
            else
                x1 = x0 + 1
            end if

            !內插計算
            Z0 = xData(k,x0)
            Z1 = xData(k,x1)
            NewData(k,i) = (Z1-Z0)*(xNewLoc(i)/dx-(x0-1))+Z0
        end do
    end do
    !$omp end parallel do
    return
    end subroutine

    subroutine interp1D_nonequal(lenx,x_input,N,x_output,inData,outData) bind(c,name='interp1D_nonequal')
    implicit none
    real(c_double) :: Z0, Z1
    integer(c_int) :: x0, x1
    integer(c_int) :: k, mid
    integer(c_int),intent(in), value :: lenx, N
    real(c_double),intent(in) :: x_output(N), inData(lenx), x_input(lenx)
    real(c_double),intent(out) :: outData(N)

    !$omp parallel do private(k,x0,x1,mid,Z0,Z1)
    do k = 1, N
        ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
        if (x_output(k) == 1.7D308) then
            outData(k) = 1.7D308
            cycle
        end if
            ! bisection search
        x0 = 1
        x1 = lenx
        do while ((x1 - x0) > 1)
            mid = (x0 + x1)/ 2
            if ((x_output(k) - x_input(mid))*(x_output(k) - x_input(x0)) <= 0) then
              x1 = mid 
            else
              x0 = mid
            end if
        end do
        Z0 = inData(x0)
        Z1 = inData(x1)
        outData(k) = (Z1-Z0)*((x_output(k)-x_input(x0))/(x_input(x1)-x_input(x0)))+Z0
    end do
    !$omp end parallel do
    return
    end subroutine

    subroutine interp1D_nonequal_layers(lenx,x_input,layers,N,x_output,inData,outData) bind(c,name='interp1D_nonequal_layers')
    implicit none
    real(c_double) :: Z0, Z1
    integer(c_int) :: x0, x1
    integer(c_int) :: k, s, mid
    integer(c_int),intent(in), value :: lenx, N, layers
    real(c_double),intent(in) :: x_output(N), inData(layers, lenx), x_input(lenx)
    real(c_double),intent(out) :: outData(layers, N)

    !$omp parallel do private(k,x0,x1,mid,Z0,Z1)
    do s = 1, layers
        do k = 1, N
            ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
            if (x_output(k) == 1.7D308) then
                outData(s,k) = 1.7D308
                cycle
            end if
            
            ! bisection search
            x0 = 1
            x1 = lenx
            do while ((x1 - x0) > 1)
                mid = (x0 + x1)/ 2
                if ((x_output(k) - x_input(mid))*(x_output(k) - x_input(x0)) <= 0) then
                  x1 = mid 
                else
                  x0 = mid
                end if
            end do
            Z0 = inData(s,x0)
            Z1 = inData(s,x1)
            outData(s,k) = (Z1-Z0)*((x_output(k)-x_input(x0))/(x_input(x1)-x_input(x0)))+Z0
        end do
    end do
    !$omp end parallel do
    return
    end subroutine



end module


