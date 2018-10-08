#! /usr/bin/env python
import sys
sys.path.insert(0, sys.path[0]+'/waf_tools')

import os
import limbo_main
import limbo

def options(opt):
    limbo_main.options(opt)
    opt.load('opencv')
    opt.load('robot_dart')

def configure(conf):
    limbo_main.configure(conf)
    conf.load('opencv')
    conf.load('robot_dart')
    
    conf.check_opencv()
    conf.check_robot_dart()

    conf.env.LIB_THREADS = ['pthread']

def shutdown(ctx):
    limbo_main.shutdown(ctx)
    
def build(bld):
    bld.recurse('src/')
