// Copyright (c) 2014 Max Planck Society

/*!@file
 * @author  Simon Bartels <sbartels@tuebingen.mpg.de>
 *
 * @date    2015-10-27
 *
 * @details 
 * This class implements the Matlab GPInnovationLocal functionality.
 *
 */

#ifndef INCLUDE_GP_INNOVATION_LOCAL_H
#define INCLUDE_GP_INNOVATION_LOCAL_H

#include <Eigen/Dense>
#include "gp.h"
#include "MathTools.hpp"
#include <cmath>
#include <iostream>

/**
 * Class that wraps the efficient_innovation function from the Matlab implementation as an object. */
class GPInnovationLocal {

public:
	/**
	 * Reference to zbel. Public for compatibility reasons.
	 */
	// const Eigen::MatrixXd zbel;

	/** 
	 * Constructor for the innovation function. Basically performs the same operations as GP_innovation_local.
	 * @param gp pointer to a Gaussian process (libGP)
	 * @param zbel an NxD Matrix where D is the input dimensionality 
	 */
	GPInnovationLocal(libgp::GaussianProcess * gp, const Eigen::MatrixXd & zbel);

	/**
	 * Destructor. Frees the heap memory allocated for kbX.
	 */
	~GPInnovationLocal();

	/**
	 * Contains the efficient_innovation function from the Matlab implementation. The main difference is that the
	 * output arguments are here part of the inputs. This saves memory allocation time.
	 * NOTE: Make sure Lx and dLxdx have been initialized properly with according resize() calls.
	 * @param x the input argument
	 * @param Lx output argument, the innovation
	 * @param dLxdx output argument, the derivative
	 */
	void efficient_innovation(const Eigen::VectorXd & x, Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::MatrixXd> dLxdx);

	void update_variables(libgp::GaussianProcess * gp, Eigen::MatrixXd zbel);

protected:
	// Eigen::MatrixXd cKkXb;
	Eigen::MatrixXd zbel;
	libgp::GaussianProcess * gp;

	double sn2; // noise

};
#endif
