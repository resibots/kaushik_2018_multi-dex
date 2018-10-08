#! /usr/bin/env python
# Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty opencv detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
	pass


@conf
def check_opencv(conf):
    includes_check = ['/usr/local/include', '/usr/include']
    libs_check = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu/']
    
    incl = ''
    try:
        conf.start_msg('Checking for OpenCV2 C++ includes (optional)')
        res = conf.find_file('opencv2/opencv.hpp', includes_check)
        res = conf.find_file('opencv2/ml/ml.hpp', includes_check)
        incl = res[:-len('opencv2/ml/ml.hpp')-1]
        conf.end_msg(incl)
    except:
        conf.end_msg('Not found in %s' % str(includes_check), 'YELLOW')
        return 1
    conf.start_msg('Checking for OpenCV2 C++ libs (optional)')
    lib_path = ''
    for lib in ['libopencv_core.so', 'libopencv_ml.so']:
        try:
            res = conf.find_file(lib, libs_check)
            lib_path = res[:-len(lib)-1]
        except:
            continue
    if lib_path == '':
        conf.end_msg('Not found in %s' % str(libs_check), 'YELLOW')
        return 1
    else:
        conf.end_msg(lib_path)
        conf.env.INCLUDES_OPENCV = [incl]
        conf.env.LIBPATH_OPENCV = [lib_path]
        conf.env.DEFINES_OPENCV = ['USE_OPENCV']
        conf.env.LIB_OPENCV = ['opencv_core', 'opencv_ml']
    return 1