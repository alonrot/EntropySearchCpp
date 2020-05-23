// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __OPTIMIZER__
#define __OPTIMIZER__

#include "gp.h"

namespace libgp
{

  /** Optimizer base class.
   *  @author Simon Bartels */
  class Optimizer
  {
    public:
      /** Constructor. */
      Optimizer() {};

      /** Destructor. */
      virtual ~Optimizer() {};

      /** Optimize a certain Gaussian process
       *  @param gp the Gaussian process
       *  @param n the maximal number of calls to gp->set_loghyper()
       *  @param verbose whether to print debug output or not */
      virtual void maximize(GaussianProcess * gp, size_t n=100, bool verbose=1) = 0;
  };

}

#endif /* __OPTIMIZER__ */

