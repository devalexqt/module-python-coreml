from distutils.core import setup, Extension
import numpy

module1 = Extension('mycoreml',sources = ['mycoreml.m',"helpers/acoreml_helper_feature_provider.m"],include_dirs=["helpers",numpy.get_include()], extra_link_args=["-framework","AppKit","-framework","Foundation",  "-framework","Accelerate", "-framework","CoreML"])#"appkit","accelerate","foundation","coreml"
# Extension('foo', ['foo.c'], include_dirs=['include'])
# Extension(...,libraries=['gdbm', 'readline'])

setup (name = 'mycoreml',
       version = '1.0',
       description = 'This is a mycoreml package',
       ext_modules = [module1])