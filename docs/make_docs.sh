#!/usr/bin/env bash
make clean
rm -r source/reference
rm source/tutorials/*
cd source/tutorials
ln -s ../../../notebooks/* .
cd ../..
sphinx-apidoc -o source/reference ../wfsim
rm source/reference/modules.rst
make html
