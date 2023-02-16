import matplotlib.pyplot as plt
import numpy as np

r=200
g=30
b=100

def RGB2colorcode(r, g, b, preview=True):
    '''
    將RGB轉為色碼

    Parameters
    ----------
    r : int 0 ~ 255
        紅色色碼
    g : int 0 ~ 255
        綠色色碼
    b : int 0 ~ 255
        藍色色碼
    preview : bool, optional
        是否要顯示顏色預覽圖 The default is True.

    Returns
    -------
    colorcode : str
        色碼

    '''
    colorcode = f'#{hex(r)[2:]}{hex(g)[2:]}{hex(b)[2:]}'.upper()
    if preview == True:
        plt.plot([0,1],[0,1],c=colorcode,linewidth=10)
        plt.text(0.35,0.1,colorcode,fontsize=30,color=colorcode)
        plt.text(0.05,0.8,f"R={r}",fontsize=30,color='#FF0000')
        plt.text(0.05,0.65,f"G={g}",fontsize=30,color='#00FF00')
        plt.text(0.05,0.5,f"B={b}",fontsize=30,color='#0000FF')
        plt.axis('off')
        plt.show()
    return colorcode

def colorcode2RGB(colorcode, preview=True):
    '''
    

    Parameters
    ----------
    colorcode : str
        色碼
    preview : bool, optional
        是否要顯示顏色預覽圖 The default is True.

    Returns
    -------
    r : int 0 ~ 255
        紅色色碼
    g : int 0 ~ 255
        綠色色碼
    b : int 0 ~ 255
        藍色色碼
    '''

    r = int(colorcode[1:3],16)
    g = int(colorcode[3:5],16)
    b = int(colorcode[5:7],16)
    if preview == True:
        plt.plot([0,1],[0,1],c=colorcode,linewidth=10)
        plt.text(0.35,0.1,colorcode.upper(),fontsize=30,color=colorcode)
        plt.text(0.05,0.8,f"R={r}",fontsize=30,color='#FF0000')
        plt.text(0.05,0.65,f"G={g}",fontsize=30,color='#00FF00')
        plt.text(0.05,0.5,f"B={b}",fontsize=30,color='#0000FF')
        plt.axis('off')
        plt.show()
    return r,g,b



