Description
=========
Vanilla Entropy Search synthetic example.

This repository is a high-level user interface to run [Entropy Search (ES)](http://www.jmlr.org/papers/volume13/hennig12a/hennig12a.pdf). The original code can be found [here](https://github.com/ProbabilisticNumerics/entropy-search).

	Philipp Hennig and Christian J Schuler,
	"Entropy Search for Information-Efficient Global Optimization", 
	The Journal of Machine Learning Research (JMLR),
	2012, accepted.

This code has been used to automatically tune parameters of a Linear Quadratic Regulator (LQR) for a robot arm to balance an inverted pole. See the associated publication [here](https://arxiv.org/abs/1605.01950) and a video description of the method and the results [here](https://youtu.be/TrGc4qp3pDM).

	Alonso Marco, Philipp Hennig, Jeannette Bohg, Stefan Schaal, Sebastian Trimpe,
	"Automatic LQR Tuning Based on Gaussian Process Global Optimization", 
	IEEE International Conference on Robotics and Automation (ICRA),
	2016, accepted.

Requirements
============
This package needs the Entropy Search (ES) library, which can be found [here](https://github.com/alonrot/ESlib).
It works under Matlab 2017 or higher.

Execute an example
==================
1. Clone the Entropy Search (ES) library, which can be found [here](https://github.com/alonrot/ESlib).
2. Add the path to the library ESlib permanently in your Matlab path. You can do that by running the next commands:
```Matlab
addpath '/full/path/to/ESlib/'
savepath
```
3. Call the function start_up() from the Matlab promt. This adds relevants paths to your Matlab session,  and attempts to compile some c++ functions in order to speed up the code execution. If the compilation does not suceed, the Matlab version of those functions (computationally less efficient) will be executed.
4. Several options can be chosen by the user in initialize_ES() and initialize_GP() (see "Remarks").
5. Run the script run_ES.m and wait for the search to finish.
6. Results are saved automatically under results/vanES_YYYY-M-D-m/ each time the script run_ES.m is called. If the exploration is stopped by the user (e.g., with Ctr+C), the current status of the exploration can be  checked by loading currentESstatus.mat.

Remarks
=======
* Change the input dimension (keyword DIM, in the code):
	* Change the input domain limits in initialize_ES().
	* Change the lengthscales and prior standard deviation in initialize_GP().
	* At the moment, ES runs with squared exponential (SE) kernel and or rational quadratic (RQ) kernel. The corresponding functions are covSEard, covSEard_dx_MD, and covRQard,covRQard_dx_MD, respectively.
	* The color code for the Gaussian process plots is:
    Red dots:   collected data
    Black dot:  Current estimate of the global minimum
    In 1D:
      Red line:                 posterior GP mean
      Red transparent surface:  +- two standard deviations, computed out of the posterior GP variance.
      Dashed line:              true function
    In 2D:
      Violet surface:           posterior GP mean
      Grey transparent surface: +- two standard deviations, computed out of the posterior GP variance.
	* The output structure (out) contains several relevant fields:
    out.FunEst: estimate of the location of the global minimum at each iteration
    out.GPs{k}, being k the desired iteration number: useful information related to the utilized Gaussian 
    process model, for example:
    out.GPs{k}.x: location of the collected data points
    out.GPs{k}.y: value of the collected data points
    out.GPs{k}.z_plot: locations where the posterior mean and std are calculated to be plotted.

Contact information
===================
For any questions, please, send an e-mail to: 

   alonso.marco@tuebingen.mpg.de

