import sys, os

libpath = os.path.join ( os.path.dirname (os.path.realpath (__file__)), os.pardir, 'lib')

if sys.version_info <= (3, 4):
  # imp deprecated in version 3.3, but importlib.find_loader deprecated in version 3.4
  # stick with deprecated imp for v3.3 for protection of my sanity
  # Note to self: script is already python2 incompatible anyways because of use of fstrings. Change this?
  import imp
  print("Using imp. Python version: "+sys.version)
  fp, pathname, description = imp.find_module('tractfinder', [ libpath ])
  imp.load_module('tractfinder', fp, pathname, description)

else:
  import importlib.util as imput

  print("Using importlib. Python version: "+sys.version)

  MODULE_NAME = 'tractfinder'
  MODULE_PATH = os.path.join(libpath, MODULE_NAME, '__init__.py')

  # Holy python import hell why have they done this
  spec = imput.spec_from_file_location(MODULE_NAME, MODULE_PATH)
  tractfinder = imput.module_from_spec(spec)
  sys.modules[spec.name] = tractfinder
  spec.loader.exec_module(tractfinder)
