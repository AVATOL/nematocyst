#!/usr/bin/env python
# Cross platform script to install dependencies.

import argparse
import urllib
import zipfile
import tarfile
import os
import shutil
import platform
import subprocess
import traceback

# paths to MATLAB
MAC_MATLAB_PATH = "/Applications/MATLAB_R2015b.app/bin/matlab"
WIN_MATLAB_PATH = "C:\\Program Files\\MATLAB\\R2015b\\bin\\matlab.exe"
LINUX_MATLAB_PATH = "/usr/local/apps/bin/matlab"

def is_mac():
    return platform.system() == 'Darwin'

def is_pc():
    return platform.system() == 'Windows'

def is_linux():
    return platform.system() == 'Linux'

def download_file(url, output):
    print "Downloading {} to {}...".format(url, output)
    urllib.urlretrieve(url, output)

def unzip_file(file, dest):
    print "Extracting zip {} to {}...".format(file, dest)
    with zipfile.ZipFile(file, 'r') as f:
        f.extractall(dest)

def untar_file(file, dest):
    print "Extracting tgz {} to {}...".format(file, dest)
    with tarfile.open(file, 'r') as f:
        f.extractall(dest)

def move_file(src, dest):
    shutil.move(src, dest)

def execute_command(application, name):
    print "Executing command {}".format(application)
    try:
        exit_status = subprocess.call(application)
    except:
        print "{} unsuccessful.".format(name)
        traceback.print_exc()
        exit(1)

    if exit_status != 0:
        print "Nonzero exit status for {}.".format(name)
        exit(1)

def cd(path):
    try:
        os.chdir(path)
    except:
        print "cd {} unsuccessful".format(path)
        traceback.print_exc()
        exit(1)

def download_dependencies():
    # LIBLINEAR
    download_file("https://github.com/cjlin1/liblinear/archive/v210.zip", "external/liblinear.zip")
    unzip_file("external/liblinear.zip", "external/")
    move_file("external/liblinear-210", "external/liblinear")

    # LIBSVM
    download_file("https://github.com/cjlin1/libsvm/archive/v320.zip", "external/libsvm.zip")
    unzip_file("external/libsvm.zip", "external/")
    move_file("external/libsvm-320", "external/libsvm")

    # EIGEN
    download_file("http://bitbucket.org/eigen/eigen/get/3.2.1.tar.gz", "external/eigen.tar.gz")
    untar_file("external/eigen.tar.gz", "external/")
    move_file("external/eigen-eigen-6b38706d90a9", "external/Eigen")

    # VLFEAT
    download_file("http://www.vlfeat.org/download/vlfeat-0.9.18-bin.tar.gz", "external/vlfeat.tar.gz")
    untar_file("external/vlfeat.tar.gz", "external/")
    move_file("external/vlfeat-0.9.18", "external/vlfeat")

    # VOWPAL WABBIT
    download_file("https://github.com/JohnLangford/vowpal_wabbit/archive/7.7.tar.gz", "external/vowpal_wabbit.tar.gz")
    untar_file("external/vowpal_wabbit.tar.gz", "external/")
    move_file("external/vowpal_wabbit-7.7", "external/vowpal_wabbit")

def install_dependencies():
    if is_mac() or is_linux():
        # LIBLINEAR
        cd('external/liblinear')
        execute_command(['make'], 'make liblinear')
        cd('../..')

        # LIBSVM
        cd('external/libsvm')
        execute_command(['make'], 'make libsvm')
        cd('../..')

        # VOWPAL WABBIT
        cd('external/vowpal_wabbit')
        execute_command(['./autogen.sh'], 'autogen')
        execute_command(['make'], 'make vowpal_wabbit')
        cd('../..')

def compile_matlab_mex(matlab_path):
    # LIBLINEAR
    cd('external/liblinear/matlab')
    func = "try;{0};catch exception;disp(getReport(exception));exit(1);end;exit".format('make')
    application = []
    if is_mac() or is_linux():
        application = [
            matlab_path,
            "-nodisplay",
            '-r "{0}"'.format(func)
        ]
    elif is_pc():
        application = [
            matlab_path,
            "-nosplash",
            "-wait",
            "-nodesktop",
            "-minimize",
            "-r",
            '"{0}"'.format(func)
        ]
    execute_command(application, 'MATLAB LIBLINEAR')
    cd('../../..')

    # LIBSVM
    cd('external/libsvm/matlab')
    func = "try;{0};catch exception;disp(getReport(exception));exit(1);end;exit".format('make')
    application = []
    if is_mac() or is_linux():
        application = [
            matlab_path,
            "-nodisplay",
            '-r "{0}"'.format(func)
        ]
    elif is_pc():
        application = [
            matlab_path,
            "-nosplash",
            "-wait",
            "-nodesktop",
            "-minimize",
            "-r",
            '"{0}"'.format(func)
        ]
    execute_command(application, 'MATLAB LIBSVM')
    cd('../../..')

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Set up all dependencies.')
    parser.add_argument('--matlab')
    args = parser.parse_args()

    # determine MATLAB path
    if is_mac():
        matlab_path = MAC_MATLAB_PATH
    elif is_pc():
        matlab_path = WIN_MATLAB_PATH
    elif is_linux():
        matlab_path = LINUX_MATLAB_PATH
    else:
        print "Unrecognized OS/Platform: {0}".format(platform.system())
        exit(1)

    if args.matlab is not None:
        matlab_path = args.matlab
    if not os.path.exists(matlab_path):
        print "Cannot find MATLAB: {}".format(matlab_path)
        exit(1)

    # make external directory
    if not os.path.exists("external"):
        os.makedirs("external")

    # download dependencies
    download_dependencies()

    # install dependencies
    install_dependencies()

    # compile MATLAB mex files
    compile_matlab_mex(matlab_path)

    # make HC-Search
    if not is_pc():
        execute_command(['make'], 'make HC-Search')

if __name__ == '__main__':
    main()
