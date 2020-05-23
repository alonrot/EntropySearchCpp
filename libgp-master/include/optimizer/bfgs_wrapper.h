#ifndef __BFGS_WRAPPER__
#define __BFGS_WRAPPER__

#include "optimizer/optimizer.h"
#include "optimizer/bfgs_optimizer.h"

namespace libgp {
class BFGS : public libgp::Optimizer {
 public:

  /*!
   * BFGS ctor
   *
   * Default constructor.
   */
  BFGS() : Optimizer() {};


  /** Destructor. */
  virtual ~BFGS() {};

  void maximize(libgp::GaussianProcess * gp, size_t n=100, bool verbose=1);

 private:
  /** The wrapper class for the GP likelihood. */
  class LLH : public bfgs_optimizer::ObjectiveFunction {
    public:
      /** Default constructor. */
      LLH(GaussianProcess * gp) : gp(gp) {};

      /** Destructor. */
      virtual ~LLH() {};

      void evaluate(const Eigen::VectorXd& x,
	                  double* function_value,
	                  Eigen::VectorXd* derivative) const;
    private:
      GaussianProcess * gp;
  };
};
}  // namespace libgp



#endif /* __BFGS_WRAPPER__ */
