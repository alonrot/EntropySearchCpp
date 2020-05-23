#include "simple_fun_wrapper.h"

void SimpleFunWrapper::evaluate(const Eigen::VectorXd& x, 
																double* function_value, 
																Eigen::VectorXd* derivative) const{

	// Don't we have to implement the contructor?
	// Don't we have to allocate memory for the pointer to sf?
	double fval = sf->get_value(x(0));
	Eigen::VectorXd fder = Eigen::VectorXd(1);
	fder(0) = sf->get_gradient(x(0));
	*function_value = fval;
	*derivative     = fder;

}