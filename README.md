# 無標題

# 安裝

### 環境檢查

使用前請確保您的環境已安裝以下套件:

numpy scipy wrf-python netCDF4 json (opencv)

| numpy | scipy | wrf-python | netCDF4 | python |
| --- | --- | --- | --- | --- |

註: opencv可選用安裝，若未安裝則無法使用meteotools.pictureprocess.movie_make。

使用前請確保您的系統已安裝以下軟體:

gfortran [https://www.mingw-w64.org/downloads/#mingw-builds](https://www.mingw-w64.org/downloads/#mingw-builds)

將gfortran加入PATH，使在指令視窗下可直接輸入gfortran即找到此編譯器。

請參考: [安裝編譯器並將編譯器加入系統路徑(PATH)](https://www.notion.so/PATH-1f16f2851fb18054b4a5c8f0ada1aee1?pvs=21)

本套件是在以下環境測試:

| 名稱 | 版本 |
| --- | --- |
| python | 3.8.12 |
| numpy | 1.24.4 |
| scipy | 1.9.1 |
| nctCDF4 | 1.6.4 |
| wrf-python | 1.3.4.1 |
| gfortran 64-bit | 14.2.0 |
| Windows 11 64-bit |  |

# 建置

下載meteotools後，請先至meteotools資料夾內執行一次python build.py，將依照您的作業系統編譯套件需要的函式庫。

請確保gfortran已安裝至您的系統，並將gfortran加入PATH，並可直接由指令呼叫。

# 更多

關於這個套件的更多資訊，請參照:

[https://www.notion.so/PATH-1f16f2851fb18054b4a5c8f0ada1aee1?pvs=4](https://www.notion.so/PATH-1f16f2851fb18054b4a5c8f0ada1aee1?pvs=21)