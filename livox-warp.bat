@echo off
rem Run the viewer straight from a checkout, after `maturin develop --release` has put the
rem extension module in python\livox_warp\. Once the package is installed (`pip install .`)
rem the `livox-warp` console script is the normal entry point instead.
rem Extra arguments pass through, e.g.  livox-warp --sim
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"
python -m livox_warp %*
