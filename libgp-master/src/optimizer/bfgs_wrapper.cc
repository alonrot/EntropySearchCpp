#include "optimizer/bfgs_wrapper.h"


namespace libgp {
	void BFGS::maximize(libgp::GaussianProcess * gp, size_t n, bool verbose){
	  LLH f(gp);
	  bfgs_optimizer::BFGS bfgs(&f, n);
    Eigen::VectorXd params = bfgs.minimize(gp->covf().get_loghyper());
    gp->covf().set_loghyper(params);
	}

	void BFGS::LLH::evaluate(const Eigen::VectorXd& x, double* function_value, Eigen::VectorXd* derivative) const{
	  gp->covf().set_loghyper(x);
	  *function_value = -gp->log_likelihood();
	  gp->log_likelihood_gradient(*derivative);
    *derivative = -(*derivative);
	}
}  // namespace libgp
