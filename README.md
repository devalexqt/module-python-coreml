# This is Python bindings for Apple CoreML
This is module enable to fast call CoreML inference in python code. It also accept precompiled models.

This project show how to implement custom module for python. But looks like we not get any benefits from using numpy in c code, because it's has it's own internal structure and native c code is faster if we working in c env. So, basically NO real reason to implent wrapper for everithing, it's just for quick prototyping and testing. If we want fast runtime we need to implement it in c code directly without numpy.

Also, this module can be used for fast test of Apple frameworks (Metal, ANE, etc...) as it's very easy to build and push data from python.

To start using/build module we need to install arm64 M1 version of python 3.10 and numpy (from website) on the host drectly!(not in virtual env)

```
pip3.10 install numpy
```

Extending Python with C or C++
https://docs.python.org/3/extending/extending.html


Building C and C++ Extensions
https://docs.python.org/3/extending/building.html#building

How to extend NumPy
https://numpy.org/doc/stable/user/c-info.how-to-extend.html#dealing-with-array-objects


## build module
```
python3 setup.py build
```

## install module
```
python3 setup.py install
```

## run and test module
```
python3 test.py
```
