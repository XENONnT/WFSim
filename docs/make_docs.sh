#!/usr/bin/env bash
make clean
rm -r source/reference
rm source/tutorials/*
cd source/tutorials
#for notebook in Understanding_WFSim_P1.ipynb Understanding_WFSim_P2.ipynb Advanced_tricks.ipynb;
#  do
#    ln -s ../../../notebooks/$notebook .
#  done
ln -s ../../../notebooks/* .
cd ../..
sphinx-apidoc -o source/reference ../wfsim
rm source/reference/modules.rst
make html