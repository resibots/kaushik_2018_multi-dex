#!/usr/bin/env python
import sys
sys.path.insert(0, './waf_tools')

VERSION = '0.0.1'
APPNAME = 'limbo'

srcdir = '.'
blddir = 'build'

import glob
import os
import subprocess
import limbo
import inspect
from waflib import Logs
from waflib.Build import BuildContext

def options(opt):
        opt.load('compiler_cxx boost waf_unit_test')
        opt.load('compiler_c')
        opt.load('eigen')
        opt.load('tbb')
        opt.load('mkl')
        opt.load('sferes')
        opt.load('limbo')
        opt.load('openmp')
        opt.load('nlopt')
        opt.load('libcmaes')
        opt.load('xcode')
        limbo.add_create_options(opt)
        opt.add_option('--oar', type='string', help='config file (json) to submit to oar', dest='oar')

def configure(conf):
        conf.load('compiler_cxx boost waf_unit_test')
        conf.load('compiler_c')
        conf.load('eigen')
        conf.load('tbb')
        conf.load('sferes')
        conf.load('openmp')
        conf.load('mkl')
        conf.load('xcode')
        conf.load('nlopt')
        conf.load('libcmaes')

        native_flags = "-march=native"
        if conf.env.CXX_NAME in ["icc", "icpc"]:
            common_flags = "-Wall -std=c++11"
            opt_flags = " -O3 -xHost -g"
            native_flags = "-mtune=native -unroll -fma"
        else:
            if conf.env.CXX_NAME in ["gcc", "g++"] and int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 47:
                common_flags = "-Wall -std=c++0x"
            else:
                common_flags = "-Wall -std=c++11"
            if conf.env.CXX_NAME in ["clang", "llvm"]:
                common_flags += " -fdiagnostics-color"
            opt_flags = " -O3 -g"

        native = conf.check_cxx(cxxflags=native_flags, mandatory=False, msg='Checking for compiler flags \"'+native_flags+'\"')
        if native:
            opt_flags = opt_flags + ' ' + native_flags
        else:
            Logs.pprint('YELLOW', 'WARNING: Native flags not supported. The performance might be a bit deteriorated.')

        conf.check_boost(lib='serialization filesystem \
            system unit_test_framework program_options \
            thread', min_version='1.39')
        conf.check_eigen()
        conf.check_tbb()
        conf.check_sferes()
        conf.check_openmp()
        conf.check_mkl()
        conf.check_nlopt()
        conf.check_libcmaes()

        conf.env.INCLUDES_LIMBO = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "deps/limbo/src"

        all_flags = common_flags + opt_flags
        conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split(' ')
        Logs.pprint('NORMAL', 'CXXFLAGS: %s' % conf.env['CXXFLAGS'])

def shutdown(ctx):
    if ctx.options.oar:
        limbo.oar(ctx.options.oar)

def build(bld):
    bld.recurse('src/')
