#import fastcompute
class DimensionError(Exception):
    '''
    陣列維度錯誤
    '''
    pass


class UnitError(Exception):
    '''
    單位錯誤
    '''
    pass


class InputError(Exception):
    '''
    輸入的資訊(如變數)錯誤
    '''
    pass


class LengthError(Exception):
    '''
    陣列元素個數錯誤
    '''
    pass
