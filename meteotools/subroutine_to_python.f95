subroutine interp2D_fast(lenx,leny,dx,dy,N,xNewLoc,yNewLoc,xyData,RThetaData)
implicit none
real*8 :: Z00, Z01, Z10, Z11, tmpZ0, tmpZ1
integer :: x0, y0, x1, y1
integer :: k,j,i
integer,intent(in) :: lenx, leny, N   ! 分別表示直角座標x、y、z，圓柱座標theta, r
real*8,intent(in) :: dx,dy
real*8,intent(in) :: xNewLoc(N), yNewLoc(N), xyData(leny,lenx)
real*8,intent(out) :: RThetaData(N)

!$omp parallel do private(i,x0,y0,x1,y1,Z00,Z10,Z01,Z11,tmpZ0,tmpZ1)
do i = 1, N
    ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
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
end subroutine

subroutine interp2D_fast_layers(lenx,leny,layers,dx,dy,N,xNewLoc,yNewLoc,xyData,RThetaData)
implicit none
real*8 :: Z00, Z01, Z10, Z11, tmpZ0, tmpZ1, xtmp, ytmp
integer :: x0, y0, x1, y1
integer :: k,i
integer,intent(in) :: lenx, leny, N, layers   ! 分別表示直角座標x、y、z，圓柱座標theta, r
real*8,intent(in) :: dx,dy
real*8,intent(in) :: xNewLoc(N), yNewLoc(N), xyData(layers,leny,lenx)
real*8,intent(out) :: RThetaData(layers,N)

!$omp parallel do private(k,i,x0,y0,x1,y1,Z00,Z10,Z01,Z11,tmpZ0,tmpZ1)
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


subroutine interp1D_fast(lenx,dx,Nr,xNewLoc,xData,RThetaData)
implicit none
real*8 :: Z0, Z1
integer :: x0, x1
integer :: i
integer,intent(in) :: lenx, Nr   ! 分別表示直角座標x、y、z，圓柱座標theta, r
real*8,intent(in) :: dx
real*8,intent(in) :: xNewLoc(Nr), xData(lenx)
real*8,intent(out) :: RThetaData(Nr)

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


subroutine interp1D_fast_layers(lenx,layers,dx,Nr,xNewLoc,xData,NewData)
implicit none
real*8 :: Z0, Z1
integer :: x0, x1
integer :: i,k
integer,intent(in) :: lenx, Nr,layers   ! 分別表示直角座標x、y、z，圓柱座標theta, r
real*8,intent(in) :: dx
real*8,intent(in) :: xNewLoc(Nr), xData(layers,lenx)
real*8,intent(out) :: NewData(layers,Nr)


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


subroutine interp3D_fast(lenx,leny,lenz,dx,dy,dz,N,xNewLoc,yNewLoc,zNewLoc,xyzData,NewData)
implicit none
real*8 :: w000, w001, w010, w011, w100, w101, w110, w111
real*8 :: tmpw00, tmpw10, tmpw01, tmpw11, tmpw0, tmpw1
real*8 :: xfrac, yfrac, zfrac
integer :: x0, y0, z0, x1, y1, z1
integer :: i
integer,intent(in) :: lenx, leny, lenz, N   ! 分別表示直角座標x、y、z，新的內插位置數目
real*8,intent(in) :: dx, dy, dz
real*8,intent(in) :: xNewLoc(N),yNewLoc(N),zNewLoc(N), xyzData(lenz, leny, lenx)
real*8,intent(out) :: NewData(N)

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

subroutine interp3D_fast_layers(lenx,leny,lenz,layers,dx,dy,dz,N,xNewLoc,yNewLoc,zNewLoc,xyzData,NewData)
implicit none
real*8 :: w000, w001, w010, w011, w100, w101, w110, w111
real*8 :: tmpw00, tmpw10, tmpw01, tmpw11, tmpw0, tmpw1
real*8 :: xfrac, yfrac, zfrac
integer :: x0, y0, z0, x1, y1, z1
integer :: k,i
integer,intent(in) :: lenx, leny, lenz, N, layers   ! 分別表示直角座標x、y、z，新的內插位置數目
real*8,intent(in) :: dx, dy, dz
real*8,intent(in) :: xNewLoc(N),yNewLoc(N),zNewLoc(N), xyzData(layers, lenz, leny, lenx)
real*8,intent(out) :: NewData(layers, N)

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


!program test
!implicit none
!real*8 :: outData(1),x_output(1)
!x_output(1) = 2.6d0
!call interp1D_(5,(/1d0,2d0,3d0,4d0,5d0/),1,x_output,(/2d0,3d0,4d0,5d0,6d0/),outData)


!end program

subroutine interp1D(lenx,x_input,N,x_output,inData,outData)
implicit none
real*8 :: Z0, Z1
integer :: x0, x1
integer :: i, k
integer,intent(in) :: lenx, N
real*8,intent(in) :: x_output(N), inData(lenx), x_input(lenx)
real*8,intent(out) :: outData(N)

!$omp parallel do private(k,x0,x1,i,Z0,Z1)
do k = 1, N
    ! 處理xNewLoc為無效值的情況，直接輸出無效值，必跳過以下的內插步驟
    if (x_output(k) == 1.7D308) then
        outData(k) = 1.7D308
        cycle
    end if
    do i = 1, lenx-1
        if ((x_output(k) - x_input(i))*(x_output(k) - x_input(i+1)) <= 0) then
            x0 = i
            x1 = i+1
            exit
        end if
    end do
    Z0 = inData(x0)
    Z1 = inData(x1)
    outData(k) = (Z1-Z0)*((x_output(k)-x_input(i))/(x_input(i+1)-x_input(i)))+Z0
end do
!$omp end parallel do
return
end subroutine

!f2py -m fastcomputecore --fcompiler=gfortran --f90flags='-fopenmp' -c subroutine_to_python.f95
!f2py -m fastcomputecore --f90flags="-fopenmp -lgomp" -c subroutine_to_python.f95
!f2py -m fastcomputecore -c subroutine_to_python.f95 --opt=-O3