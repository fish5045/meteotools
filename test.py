

class t:
    def __init__(self):
        self.a = 1
        self.b = 2

    def abc(self):
        c = 1
        print(dir(self))
        print('a', 'b' in dir(self))


tt = t()
tt.abc()
