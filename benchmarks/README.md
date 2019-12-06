# Benchmarks

cvxpnpl was benchmarked on both both synthetic and real environments, stand-alone and versus a couple of competitor methods. Due to the sheer number of optional 3rd party dependencies, some not exactly redistributable friendly, this section of the repo has its own independent requirements and setup steps.

The folder structure is organized as follows:
- `null`: produces results for the null space search alternative strategy arguing on the value of formulating the problem as an SDP. Produces the plots in Figure 4;
- `rc`: produces results for the discussion on the inclusion of the redundant constraint. Produces the plots in Figure 3;
- `scalability`: evaluates the execution runtime scalability with an increasing number of points. Produces the plots in Figure 5;
- `synth`: compares `cvxpnpl` against other methods. Produces the plots in Figure 6;
- `toolkit`: a supporting package which includes common routines and support classes used by all benchmarks.

## Table of Contents

1. [3rd Party Setup](#3rd-party-setup)
	1. [Installing OpenGV](#installing-opengv)
	2. [Installing the Companion Toolbox from Vakhitov et al. 2016](#installing-the-companion-toolbox-from-vakhitov-et-al-2016)
		1. [Setting up the Python engine for MATLAB](#setting-up-the-python-engine-for-matlab)
		2. [Installing the PnPL Toolbox](http://localhost:6419/#installing-the-pnpl-toolbox)
2. [Running the Benchmarks](#running-the-benchmarks)

<!-- ## Synthetic Data

The synthetic data experiments validate cvxpnpl against of numerous random realizations of points and lines, as well as other methods. These benchmarks generate the supporting data for the plots displayed in the paper.

**Important Note:** Until I figure out a way to dynamically import all the optional modules required by the competitor methods, you are required to meet all dependencies in order to run the benchmarks. However, in case you're not interested in going through all that trouble, remember that **you can always comment out the unavailable 3rd party methods and bypass the need for these dependencies**. It's not the most optimal solution but it will be a valuable workaround for most.
 -->
## 3rd Party Setup

I only considered mandatory dependencies, packages which are being distributed through PyPi/pip. Everything else is considered optional. Each benchmark will evaluate at runtime if all dependencies are met and warn the user is something is missing.

1. Install all pip dependencies. From the root folder invoke
	```
	cd benchmarks
	pip install -r requirements.txt
	```
2. Compile and install the python bindings for [OpenGV](https://github.com/laurentkneip/opengv). [More detailed info here](#installing-opengv).
3. Install the companion [toolbox](https://github.com/alexandervakhitov/pnpl) from [Vakhitov et al. 2016](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_36). [More detailed info here](#installing-the-companion-toolbox-from-vakhitov-et-al-2016).

### Installing OpenGV

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

### Installing the Companion Toolbox from Vakhitov et al. 2016

This [MATLAB toolbox](https://github.com/alexandervakhitov/pnpl), which accompanied the main [paper](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_36), implements a number of useful PnP, PnL and PnPL methods. As you might have guessed, we are now in MATLAB territory. Using this toolbox requires you to have a somewhat recent MATLAB installation. The interface between MATLAB and Python is done through MATLAB's Python engine and each MATLAB version exports a Python engine compiled against a few versions of Python. I.e. **the version of MATLAB you're using will place a requirement on the Python versions you can use**.

```
$ export MATLABPATH="$(find /usr/stud/agostinh/Development/3rdparty/pnpl -type d -follow -print | tr '\n' ':')"
````

#### Setting up the Python engine for MATLAB

1. Find your MATLAB root folder
	```
	$ matlab -nodisplay -nojvm -r "disp(matlabroot); exit"
	```
	Unfortunately MATLAB produces a splash text so you'll have to manually write down the path it returns. I will henceforth designate this path as `MATLAB_ROOT`, so better to export it.
	```
	$ export MATLAB_ROOT="copy the path here"
	```
2. Navigate to the Python engine folder
	```
	$ cd ${MATLAB_ROOT}/extern/engines/python
	```
3. Print out the supported Python versions for your MATLAB engine.
	```
	$ python -c "from setup import _supported_versions; print(_supported_versions)"
	['2.7', '3.5', '3.6']
	```
	As you can tell, in my case it supports version 2.7, 3.5 and 3.6.
4. Install it as any other package
	```
	$ python setup.py install
	```
5. Test it
	```
	$ python -c "import matlab; print(matlab)"
	```

#### Installing the PnPL Toolbox

1. Clone the repository at https://github.com/alexandervakhitov/pnpl
2. Ensure that the toolbox is visible to MATLAB at startup. For more instructions on how to do that check [this](https://www.mathworks.com/help/matlab/matlab_env/add-folders-to-matlab-search-path-at-startup.html). Since the toolbox has a number of subfolders and all of them need to be visible on MATLAB's path, the best strategy here is to adopt a `startup.m`.
3. Open MATLAB and run
	```matlab
	path
	```
	The first folder should be the user's documents folder. Change directory to that folder.
4. Edit the startup file
	```matlab
	edit startup.m
	```
	And add the following line inside
	```matlab
	addpath(genpath('<path_to_pnpl>'));
	```
	**Don't forget to replace `<path_to>` with the correct folder**.
5. Test it
	```
	$ matlab -nodisplay -nojvm -r "help OPnP; exit"
	```
	If everything went well, it should print the help docstring.
	```
	 will polish the solution in default
	```

## Running the Benchmarks

Running the benchmarks should be performed from with **the `benchmarks` folder as the current working directory**. It made no sense properly install the auxiliary `toolkit` just because of benchmarks, so this is a small price to pay.


To run the benchmarks you just need to run the corresponding Python script and after the computations are complete, the results will be plotted. In case you wish to save the benchmark results, invoke the `--save` optional switch and to restore a previously saved session to display its results, the `--load` switch is available. For a detailed example of the command check the usage and options below

```
$ python synth/pnp.py -h
usage: pnp.py [-h] [--save SAVE | --load LOAD] [--tight | --no-display]
              [--runs RUNS]

optional arguments:
  -h, --help    show this help message and exit
  --save SAVE   File path to store the session data.
  --load LOAD   File path to load and plot session data.
  --tight       Show tight figures.
  --no-display  Don't display any figures.
  --runs RUNS   Number of runs each scenario is instantiated.
```
The same usage applies to all other scripts exported by the benchmarks.

## Real Data

