# Benchmarks

cvxpnpl was benchmarked on both both synthetic and real environments, stand-alone and versus a couple of competitor methods. Due to the sheer number of optional 3rd party dependencies, some not exactly redistributable friendly, this section to the repo has its own independent requirements and setup steps.

## Synthetic Data

The synthetic data experiments validate cvxpnpl against of numerous random realizations of points and lines, as well as other methods. These benchmarks generate the supporting data for the plots displayed in the paper.

### Setup

1. Install all pip dependencies. From the root folder invoke
  ```
  pip install -r benchmarks/requirements.txt
  ```
2. Compile and install the python bindings for [OpenGV](https://github.com/laurentkneip/opengv). [More detailed info here](#installing-opengv).

#### Installing OpenGV

In this section I will provide some guiding steps (for Unix style systems) on how to download, compile and install the Python bindings for OpenGV.

1. Clone the repo from https://github.com/laurentkneip/opengv
```
git clone --recursive https://github.com/laurentkneip/opengv
```
2. Compiling OpenGV requires CMake and Eigen3.
```
$ cd opengv
$ mkdir build install
$ cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=../install/ -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE="$(which python)"
$ make install
```
3. Copy the Python module to your local installation
```
$ cp -v ../install/lib/python*/site-packages/pyopengv.* $(python -c "import site; print(site.getsitepackages()[0])")
```
4. Test it
```
python -c "import pyopengv; print(pyopengv)"
```
### Running the Benchmarks

To run the benchmarks you just need to run the Python script and after the computations are complete, the results will be plotted. In case you wish to save the benchmark results, invoke the `--save` optional switch and to restore a previously saved session to display its results, the `--load` switch is available.

```
$ python benchmarks/pnp_synth.py -h
usage: pnp_synth.py [-h] [--save SAVE | --load LOAD]

optional arguments:
  -h, --help   show this help message and exit
  --save SAVE  File path to store the session data.
  --load LOAD  File path to load and plot session data.
```
The same usage applies to `pnp_synth.py`, `pnl_synth.py` and `pnpl_synth.py`.

## Real Data

### Setup

1. Install cvxpnpl on your system, through the methods described in [here](https://github.com/SergioRAgostinho/cvxpnpl/blob/master/README.md).
