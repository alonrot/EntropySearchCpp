Description
=========
Reimplementation of [Entropy Search (ES)](http://www.jmlr.org/papers/volume13/hennig12a/hennig12a.pdf) in `c++`. The original code can be found [here](https://github.com/ProbabilisticNumerics/entropy-search). An alternative version that improves on efficiency of the original ES code and provides several plotting tools can be found [here](https://github.com/alonrot/userES).

	Philipp Hennig and Christian J Schuler,
	"Entropy Search for Information-Efficient Global Optimization", 
	The Journal of Machine Learning Research (JMLR),
	2012, accepted.

This code has been used to automatically tune parameters of a whole-body Linear Quadratic Regulator (LQR) for a two-legged humanoid robot to perform a squatting task. See the associated publication [here](https://arxiv.org/abs/1605.01950) and a video description of the method and the results [here](https://youtu.be/udJAK60IWEc).

	Alonso Marco, Philipp Hennig, Jeannette Bohg, Stefan Schaal, Sebastian Trimpe,
	"Automatic LQR Tuning Based on Gaussian Process Global Optimization", 
	IEEE International Conference on Robotics and Automation (ICRA),
	2016, accepted.

Requirements
============
This package requires:
* [yaml-cpp](https://codedocs.xyz/jbeder/yaml-cpp.svg): [YAML 1.2 spec](http://www.yaml.org/spec/1.2/spec.html) parser and emitter in C++.
* [libgp](https://github.com/mblum/libgp): C++ library for Gaussian process regression
* [cmake](http://www.cmake.org/): cross-platform, open-source build system
* [Eigen3](http://eigen.tuxfamily.org/): template library for linear algebra

Compatible versions of `yaml-cpp` and `libgp` are included along with entropy_search (this work).

Installation
============
TODO

Example
=======
TODO

Contact information
===================
For any questions, please, send an e-mail to: 

   alonso.marco(at)tuebingen.mpg.de

