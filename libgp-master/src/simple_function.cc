#include "simple_function.h"

SimpleFunction::SimpleFunction(Eigen::VectorXd param)
{
	this->param = param;
	this->n_par = param.size();
	this->current_max = 0;
}

SimpleFunction::~SimpleFunction()
{
}

double
SimpleFunction::get_value(double x)
{

	double y = 0;

	for(int i = 0; i < n_par; i++)
		y += pow( x , i )*param(i);

	return y;

}

double
SimpleFunction::get_gradient(double x)
{

	double G = 0;

	for(int i = 1; i < n_par; i++)
		G += i*pow( x , i - 1 )*param(i);

	return G;
}


double
SimpleFunction::get_hessian(double x)
{

	double H = 0;

	for(int i = 2; i < n_par; i++)
		H += i*(i-1)*pow( x , i - 2 )*param(i);

	return H;
}


