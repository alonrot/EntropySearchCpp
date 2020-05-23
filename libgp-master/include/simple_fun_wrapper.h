#ifndef __SIMPLE_FUN_WRAPPER_H__
#define __SIMPLE_FUN_WRAPPER_H__

#include "optimizer/bfgs_optimizer.h"
#include "simple_function.h"

class SimpleFunWrapper : public bfgs_optimizer::ObjectiveFunction
{

public:

	// Constructor: inline initialization of class members
	SimpleFunWrapper(SimpleFunction * sf) : sf(sf) {};

	~SimpleFunWrapper() {};

  void evaluate(const Eigen::VectorXd& x,
                double* function_value,
                Eigen::VectorXd* derivative) const;

private:
	SimpleFunction * sf;

};

#endif /* __SIMPLE_FUN_WRAPPER_H__ */