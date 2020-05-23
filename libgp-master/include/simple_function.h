#ifndef __SIMPLE_FUNCTION_H__
#define __SIMPLE_FUNCTION_H__

#include <cmath>
#include <Eigen/Dense>
#include <iostream>

class SimpleFunction
{

public:
	SimpleFunction(Eigen::VectorXd param);

	~SimpleFunction();

	double get_value(double x);

	double get_gradient(double x);

	double get_hessian(double x);

	double current_max;

protected:
	int n_par;
	Eigen::VectorXd param;

};

#endif /* __SIMPLE_FUNCTION_H__ */