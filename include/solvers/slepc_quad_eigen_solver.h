// The libMesh Finite Element Library.
// Copyright (C) 2002-2017 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



#ifndef LIBMESH_SLEPC_QUAD_EIGEN_SOLVER_H
#define LIBMESH_SLEPC_QUAD_EIGEN_SOLVER_H

#include "libmesh/libmesh_config.h"

#ifdef LIBMESH_HAVE_SLEPC

// Local includes
#include "libmesh/quad_eigen_solver.h"
#include "libmesh/slepc_macro.h"

// SLEPc include files.
EXTERN_C_FOR_SLEPC_BEGIN
# include <slepcpep.h>
EXTERN_C_FOR_SLEPC_END

namespace libMesh
{

/**
 * This class provides an interface to the SLEPc polynomial
 * eigenvalue solver library from http://slepc.upv.es/.
 *
 * \author Steffen Peterson (polynomial interface added by Orie Cecil 2018)
 * \date 2005
 * \brief EigenSolver implementation based on SLEPc.
 */
template <typename T>
class SlepcQuadEigenSolver : public QuadEigenSolver<T>
{

public:

  /**
   *  Constructor. Initializes Petsc data structures
   */
  SlepcQuadEigenSolver(const Parallel::Communicator & comm_in
                   LIBMESH_CAN_DEFAULT_TO_COMMWORLD);


  /**
   * Destructor.
   */
  ~SlepcQuadEigenSolver();


  /**
   * Release all memory and clear data structures.
   */
  virtual void clear() libmesh_override;


  /**
   * Initialize data structures if not done so already.
   */
  virtual void init() libmesh_override;

  /**
   * This function calls the SLEPc solver to compute
   * the eigenpairs for the quadratic eigenproblem
   * defined by the matrix_A, matrix_B, and matrix_C
   * with the problem formulated as
   * (lam^2*A + lam*B + C)*x = 0
   * which are of type SparseMatrix. The argument
   * \p nev is the number of eigenpairs to be computed
   * and \p ncv is the number of basis vectors to be
   * used in the solution procedure. Return values
   * are the number of converged eigen values and the
   * number of the iterations carried out by the eigen
   * solver.
   */
  virtual std::pair<unsigned int, unsigned int>
  solve_quadratic(SparseMatrix<T> & matrix_A,
                    SparseMatrix<T> & matrix_B,
                    SparseMatrix<T> & matrix_C,
                    int nev,
                    int ncv,
                    const double tol,
                    const unsigned int m_its) libmesh_override;

   /**
   * This function returns the real and imaginary part of the
   * ith eigenvalue and copies the respective eigenvector to the
   * solution vector. Note that also in case of purely real matrix
   * entries the eigenpair may be complex values.
   */
  virtual std::pair<Real, Real>
  get_eigenpair (dof_id_type i,
                 NumericVector<T> & solution_in) libmesh_override;

  /**
   * @returns the relative error ||A*x-lambda*x||/|lambda*x|
   * of the ith eigenpair. (or the equivalent for a general eigenvalue problem)
   */
  Real get_relative_error (unsigned int i);

  /**
   * Provide one basis vector for the initial guess
   */
  virtual void
  set_initial_space(NumericVector<T> & initial_space_in) libmesh_override;

  /**
   * Returns the raw SLEPc pep context pointer.
   */
  PEP pep() { this->init(); return _pep; }

private:

  /**
   * Helper function that actually performs the generalized eigensolve.
   */
  std::pair<unsigned int, unsigned int> _solve_quadratic_helper (Mat mat_A,
                                                                   Mat mat_B,
                                                                   Mat C,
                                                                   int nev,
                                                                   int ncv,
                                                                   const double tol,
                                                                   const unsigned int m_its);

  /**
   * Tells Slepc to use the user-specified solver stored in
   * \p _eigen_solver_type
   */
  void set_slepc_solver_type ();

  /**
   * Tells Slepc to deal with the type of problem stored in
   * \p _eigen_problem_type
   */
  void set_slepc_problem_type ();

  /**
   * Tells Slepc to compute the spectrum at the position
   * stored in \p _position_of_spectrum
   */
  void set_slepc_position_of_spectrum();

  /**
   * Eigenproblem solver context
   */
  PEP _pep;

};


/*----------------------- inline functions ----------------------------------*/
template <typename T>
inline
SlepcQuadEigenSolver<T>::SlepcQuadEigenSolver (const Parallel::Communicator & comm_in) :
  QuadEigenSolver<T>(comm_in)
{
  this->_eigen_solver_type  = TOAR;
  this->_eigen_problem_type = PEPGEN;
}



template <typename T>
inline
SlepcQuadEigenSolver<T>::~SlepcQuadEigenSolver ()
{
  this->clear ();
}

} // namespace libMesh


#endif // #ifdef LIBMESH_HAVE_SLEPC
#endif // LIBMESH_SLEPC_EIGEN_SOLVER_H
