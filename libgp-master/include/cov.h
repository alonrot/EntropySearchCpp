// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_H__
#define __COV_H__

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "sampleset.h"

namespace libgp
{

  /** Covariance function base class.
   *  @author Manuel Blum
   *  @ingroup cov_group 
   *  @todo implement more covariance functions */
  class CovarianceFunction
  {
    public:
      /** Constructor. */
      CovarianceFunction() {};

      /** Destructor. */
      virtual ~CovarianceFunction() {};

      /** Initialization method for atomic covariance functions. 
       *  @param input_dim dimensionality of the input vectors */
      virtual bool init(int input_dim) 
      { 
        return false;
      };

      /** Initialization method for compound covariance functions. 
       *  @param input_dim dimensionality of the input vectors 
       *  @param first first covariance function of compound
       *  @param second second covariance function of compound */
      virtual bool init(int input_dim, CovarianceFunction * first, CovarianceFunction * second)
      {
        return false;
      };

      virtual bool init(int input_dim, int filter, CovarianceFunction * covf)
      {
        return false;
      };

      /** Computes the covariance of two input vectors.
       *  @param x1 first input vector
       *  @param x2 second input vector
       *  @return covariance of x1 and x2 */
      virtual double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) = 0;

      /**
       * Vectorized version of the above get where K is the output matrix.*/
      virtual void get(const Eigen::Ref<const Eigen::MatrixXd> X1, const Eigen::Ref<const Eigen::MatrixXd> X2, Eigen::Ref<Eigen::MatrixXd> K){
          size_t n = X1.rows();
          size_t m = X2.rows();
          for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
              K(i, j) = get(X1.row(i), X2.row(j));
            }
          }
      };

      /**
       * Computes the cross covariance between the input points and X2. K is the output matrix. */
      virtual void getX(SampleSet * sampleset, const Eigen::Ref<const Eigen::MatrixXd> X2, Eigen::Ref<Eigen::MatrixXd> K) {
          size_t n = sampleset->size();
          size_t m = X2.rows(); //TODO: should this be cols?
          for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
              K(i, j) = get(sampleset->x(i), X2.row(j)); //TODO: does this work with row or do I have to use col?
            }
          }
      };


      /** Covariance gradient of two input vectors with respect to the hyperparameters.
       *  @param x1 first input vector
       *  @param x2 second input vector
       *  @param grad covariance gradient */
      virtual void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad) = 0;

      /**
       * Writes the derivative of k(X, x) w.r.t. x into dkXxdx, where:
          x must be a single point [D 1]
          X = [x1, x2, ..., xN]
          dkXxdx(x,xi) = d/dx{ k(x,xi) }, size(dkXxdx(x,xi)) = [D 1]
          dkXxdx = [ dkXxdx(x,x1) dkXxdx(x,x2) ... dkXxdx(x,xN) ], size(dkXxdx) = [D N]
       */
      virtual void compute_dkdx(const Eigen::VectorXd & x, const Eigen::Ref<const Eigen::VectorXd> & kXx, SampleSet * sampleset, Eigen::Ref<Eigen::MatrixXd> dkXxdx) {
          size_t n = sampleset->size();
          for(size_t i = 0; i < n; i++){
            compute_dkdx(x, kXx.row(i), sampleset->x(i), dkXxdx.col(i));
          }
      };
       
      /**
       *  Computes the derivative dkXxdx(x,xi), where:
          x must be a single point [D 1]
          kZx_row is a row of a column vector, so it is actually a scalar, i.e., k(x,xi)
          Z_col is a point of the sample set, as a column vector, i.e., xi
          dkZxdx_col is dkXxdx(x,xi)
       */
      virtual void compute_dkdx(const Eigen::VectorXd & x, const Eigen::Ref<const Eigen::VectorXd> & kZx_row, const Eigen::Ref<const Eigen::MatrixXd> & Z_col, Eigen::Ref<Eigen::MatrixXd> dkZxdx_col){
          //TODO: make this method abstract and implement it for every cov function
          //    exit(-1);
          dkZxdx_col.resize(kZx_row.size(), x.size());
          dkZxdx_col.setZero();
          std::cerr << "dkdx not implemented!" << std::endl;
      };

      /** Update parameter vector.
       *  @param p new parameter vector */
      virtual void set_loghyper(const Eigen::VectorXd &p);

      /** Update parameter vector.
       *  @param p new parameter vector */
      virtual void set_loghyper(const double p[]);

      /** Get number of parameters for this covariance function.
       *  @return parameter vector dimensionality */
      size_t get_param_dim();

      /** Get input dimensionality.
       *  @return input dimensionality */
      size_t get_input_dim();

      /** Get log-hyperparameter of covariance function.
       *  @return log-hyperparameter */
      Eigen::VectorXd get_loghyper();

      /** Returns a string representation of this covariance function.
       *  @return string containing the name of this covariance function */
      virtual std::string to_string() = 0;

      /** Draw random target values from this covariance function for input X. */
      Eigen::VectorXd draw_random_sample(Eigen::MatrixXd &X);

      bool loghyper_changed;

    protected:
      /** Input dimensionality. */
      size_t input_dim;

      /** Size of parameter vector. */
      size_t param_dim;

      /** Parameter vector containing the log hyperparameters of the covariance function.
       *  The number of necessary parameters is given in param_dim. */
      Eigen::VectorXd loghyper;

  };

}

#endif /* __COV_H__ */

/** Covariance functions available for Gaussian process models. 
 *  There are atomic and composite covariance functions. 
 *  @defgroup cov_group Covariance Functions */
