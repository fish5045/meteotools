"""
Build the equation core solver (SE_eqn_coreVx.x.f95; fortran 95 source code)
for Sawyer-Eliassen Equation Solver (SEES) by f2py.
Built module: fastcompute
Execute: python build.py

Version: 1.1
Author: Shang-En Li
Date: 20230216
"""

import os
import platform
import subprocess
import shutil
from pathlib import Path
import inspect
frame = inspect.currentframe()
file_path = frame.f_globals['__file__']
path_of_meteotools = os.path.dirname(os.path.abspath(file_path))

platform_name = platform.system()

class task:
    def __init__(self, build_name, source_name, source_version, source_type,
                 optimization='-O3'):
        self.build_name = build_name
        self.source_name = source_name
        self.source_version = source_version
        self.source_type = source_type
        self.optimization = optimization

        self.source_code = source_name

        if source_version != '':
            self.source_code += source_version
        if source_type != '':
            self.source_code += '.'+source_type

        self.platform = platform_name

        if self.platform == 'Windows':
            self.lib_extension = '.dll'
        elif self.platform == 'Linux':
            self.lib_extension = '.so'



    def UTF8_to_Big5(self):
        try:
            with open(self.source_code, 'r', encoding='UTF-8') as f:
                content = f.read()
            with open(self.source_code, 'w', encoding='Big5') as f:
                f.write(content)
        except:
            pass

    def Big5_to_UTF8(self):
        try:
            with open(self.source_code, 'r', encoding='Big5') as f:
                content = f.read()
            with open(self.source_code, 'w', encoding='UTF-8') as f:
                f.write(content)
        except:
            pass

    def check_source_encoding(self):
        if self.platform == 'Windows':
            self.UTF8_to_Big5()
        elif self.platform == 'Linux':
            self.Big5_to_UTF8()

    def move_compiled_to_destination(self):
        if self.platform == 'Windows':
            trans = str.maketrans('\\', '/')
            dllname = str(self.back_content).split(
                'copying')[-1].split('->')[0].split('\\')[-1].strip()
            dlldir = '\\'.join(str(self.back_content).split(
                'copying')[-1].split('->')[-1].split(r'\r')[0].strip().split(r'\\'))
            os.replace((dlldir+'\\'+dllname), dllname)
            shutil.rmtree(f'{build_name}')

    def run_compile(self,):
        
        compiler = 'gfortran'
        flags = '-shared -fPIC -fopenmp -pthread'
        linked_library = '-lgomp'
        output = f'{path_of_meteotools}/lib/{self.source_name}{self.lib_extension}'
        compile_command = compiler + ' ' \
                          + flags + ' ' \
                          + linked_library + ' ' \
                          + '-o ' + output + ' ' \
                          + path_of_meteotools+'/'+self.source_code + ' ' \
                          + self.optimization

        self.check_source_encoding()
        self.back_content = subprocess.check_output(
            compile_command, shell=True)


if __name__ == '__main__':
    Path('lib').mkdir(parents=True, exist_ok=True)
    build_name = 'fastcompute'
    source_name = 'fastcompute'
    source_version = ''
    source_type = 'f90'
    optimization = '-O3'
    
    fastcompute = task(build_name, source_name, source_version,
                 source_type, optimization)
    fastcompute.run_compile()
