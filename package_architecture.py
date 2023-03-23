import json
import os
from os.path import isdir
import importlib
import inspect

filters = ["__builtins__", "__cached__", "__doc__", "__file__", "__loader__",
           "__name__", "__package__", "__spec__", "np", "pj", "__all__",
           "plt", "tm", "fc", "wrf", "os", "Path", 'cv2', 'pth'
           ]


def process_py(pysource):
    pypath = pysource.split('/')
    for p in range(len(pypath)-1, -1, -1):
        if pypath[p] == '':
            pypath.remove('')
    if '.' in pypath:
        pypath.remove('.')
    pypath = '.'.join(pypath)[:-3]
    mod = importlib.import_module(pypath)

    pycontent = []
    for obj in dir(mod):
        if obj not in filters:
            if inspect.ismodule(eval(f'mod.{obj}')) == True:
                continue
            elif inspect.isfunction(eval(f'mod.{obj}')) == True:
                var = ', '.join(eval(f'mod.{obj}.__code__.co_varnames')[
                                :eval(f'mod.{obj}.__code__.co_argcount')])
                pycontent.append(f'<func> {obj}({var})')
            elif inspect.isclass(eval(f'mod.{obj}')) == True:
                if Exception in inspect.getmro(eval(f'mod.{obj}')):
                    pycontent.append(f'<exception> {obj}')
                else:
                    pycontent.append(f'<class> {obj}')
            else:
                if type(eval(f'mod.{obj}')) == str:
                    pycontent.append(f'<str> {obj}')
                elif type(eval(f'mod.{obj}')) == dict:
                    pycontent.append(f'<dict> {obj}')
                elif type(eval(f'mod.{obj}')) == set:
                    pycontent.append(f'<set> {obj}')
                elif type(eval(f'mod.{obj}')) == tuple:
                    pycontent.append(f'<tuple> {obj}')
                elif type(eval(f'mod.{obj}')) == list:
                    pycontent.append(f'<list> {obj}')
                elif type(eval(f'mod.{obj}')) == int:
                    pycontent.append(f'<int> {obj}')
                elif type(eval(f'mod.{obj}')) == float:
                    pycontent.append(f'<float> {obj}')
                elif type(eval(f'mod.{obj}')) == complex:
                    pycontent.append(f'<complex> {obj}')
                elif type(eval(f'mod.{obj}')) == bool:
                    pycontent.append(f'<bool> {obj}')
                else:
                    pycontent.append(f'<others> {obj}')

    return sorted(pycontent)


def process_dir(above_dir, d):
    content = os.listdir(above_dir)
    for file in content:
        if isdir(above_dir + '/' + file) == True:
            d[file] = dict()
            process_dir(f'{above_dir}/{file}/', d[file])
        elif f'{above_dir}/{file}'[-3:] == '.py':
            d[file] = dict()
            d[file] = process_py(f'{above_dir}/{file}/')
            print(f'{above_dir}/{file}/')


if __name__ == '__main__':
    package_dir = 'meteotools'
    package = dict()

    process_dir(package_dir, package)

    with open('architecture.json', 'w') as f:
        f.write(json.dumps(package, indent=4))
