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



#include "libmesh/libmesh_common.h"

// #if defined(LIBMESH_HAVE_SLEPC) && defined(LIBMESH_HAVE_PETSC)


// C++ includes

// Local Includes
#include "libmesh/libmesh_logging.h"
#include "libmesh/petsc_matrix.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/slepc_quad_eigen_solver.h"
// #include "libmesh/shell_matrix.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/solver_configuration.h"

namespace libMesh
{

template <typename T>
void SlepcQuadEigenSolver<T>::clear ()
{
  if (this->initialized())
    {
      this->_is_initialized = false;

      PetscErrorCode ierr=0;

      ierr = LibMeshPEPDestroy(&_pep);
      LIBMESH_CHKERR(ierr);

      // SLEPc default quadratic eigenproblem solver
      this->_eigen_solver_type = TOAR;
    }
}


template <typename T>
void SlepcQuadEigenSolver<T>::init ()
{

  PetscErrorCode ierr=0;

  // Initialize the data structures if not done so already.
  if (!this->initialized())
    {
      this->_is_initialized = true;

      // Create the eigenproblem solver context
      ierr = PEPCreate (this->comm().get(), &_pep);
      LIBMESH_CHKERR(ierr);

      // Set user-specified  solver
      set_slepc_solver_type();
    }
}

template <typename T>
std::pair<unsigned int, unsigned int>
SlepcQuadEigenSolver<T>::solve_quadratic (SparseMatrix<T> & matrix_A_in,
                                          SparseMatrix<T> & matrix_B_in,
                                          SparseMatrix<T> & matrix_C_in,
                                          int nev,                  // number of requested eigenpairs
                                          int ncv,                  // number of basis vectors
                                          const double tol,         // solver tolerance
                                          const unsigned int m_its) // maximum number of iterations
{
  this->init ();

  // Make sure the data passed in are really of Petsc types
  PetscMatrix<T> * matrix_A = dynamic_cast<PetscMatrix<T> *>(&matrix_A_in);
  PetscMatrix<T> * matrix_B = dynamic_cast<PetscMatrix<T> *>(&matrix_B_in);
  PetscMatrix<T> * matrix_C = dynamic_cast<PetscMatrix<T> *>(&matrix_C_in);

  if (!matrix_A || !matrix_B || !matrix_C)
    libmesh_error_msg("Error: inputs to solve_quadratic() must be of type PetscMatrix.");

  // Close the matrix and vectors in case this wasn't already done.
  matrix_A->close ();
  matrix_B->close ();
  matrix_C->close ();

  return _solve_quadratic_helper (matrix_A->mat(), matrix_B->mat(), matrix_C->mat(), nev, ncv, tol, m_its);
}

template <typename T>
std::pair<unsigned int, unsigned int>
SlepcQuadEigenSolver<T>::_solve_quadratic_helper (Mat mat_A,
                                                  Mat mat_B,
                                                  Mat mat_C,
                                                  int nev,                  // number of requested eigenpairs
                                                  int ncv,                  // number of basis vectors
                                                  const double tol,         // solver tolerance
                                                  const unsigned int m_its) // maximum number of iterations
{
  LOG_SCOPE("solve_quadratic()", "SlepcQuadEigenSolver");

  PetscErrorCode ierr=0;

  // converged eigen pairs and number of iterations
  PetscInt nconv=0;
  PetscInt its=0;
  PetscInt n_mat = 3;
  Mat A[3] = {mat_A, mat_B, mat_C};

#ifdef  DEBUG
  // The relative error.
  PetscReal error, re, im;

  // Pointer to vectors of the real parts, imaginary parts.
  PetscScalar kr, ki;
#endif

  // Set operators.
  ierr = PEPSetOperators (_pep, n_mat, A);
  LIBMESH_CHKERR(ierr);

  //set the problem type and the position of the spectrum
  set_slepc_problem_type();
  set_slepc_position_of_spectrum();

  // Set eigenvalues to be computed.
#if SLEPC_VERSION_LESS_THAN(3,0,0)
  ierr = PEPSetDimensions (_eps, nev, ncv);
#else
  ierr = PEPSetDimensions (_pep, nev, ncv, PETSC_DECIDE);
#endif
  LIBMESH_CHKERR(ierr);


  // Set the tolerance and maximum iterations.
  ierr = PEPSetTolerances (_pep, tol, m_its);
  LIBMESH_CHKERR(ierr);

  // Set runtime options, e.g.,
  //      -eps_type <type>, -eps_nev <nev>, -eps_ncv <ncv>
  // Similar to PETSc, these options will override those specified
  // above as long as EPSSetFromOptions() is called _after_ any
  // other customization routines.
  ierr = PEPSetFromOptions (_pep);
  LIBMESH_CHKERR(ierr);

  // // If the SolverConfiguration object is provided, use it to override
  // // solver options.
  // if (this->_solver_configuration)
  //   {
  //     this->_solver_configuration->configure_solver();
  //   }

  // Solve the eigenproblem.
  ierr = PEPSolve (_pep);
  LIBMESH_CHKERR(ierr);

  // Get the number of iterations.
  ierr = PEPGetIterationNumber (_pep, &its);
  LIBMESH_CHKERR(ierr);

  // Get number of converged eigenpairs.
  ierr = PEPGetConverged(_pep,&nconv);
  LIBMESH_CHKERR(ierr);


#ifdef DEBUG
  // ierr = PetscPrintf(this->comm().get(),
  //         "\n Number of iterations: %d\n"
  //         " Number of converged eigenpairs: %d\n\n", its, nconv);

  // Display eigenvalues and relative errors.
  ierr = PetscPrintf(this->comm().get(),
                     "           k           ||Ax-kx||/|kx|\n"
                     "   ----------------- -----------------\n" );
  LIBMESH_CHKERR(ierr);

  for (PetscInt i=0; i<nconv; i++ )
    {
      ierr = PEPGetEigenpair(_pep, i, &kr, &ki, PETSC_NULL, PETSC_NULL);
      LIBMESH_CHKERR(ierr);

#if SLEPC_VERSION_LESS_THAN(3,6,0)
      ierr = PEPComputeRelativeError(_pep, i, &error);
#else
      ierr = PEPComputeError(_pep, i, PEP_ERROR_RELATIVE, &error);
#endif
      LIBMESH_CHKERR(ierr);

#ifdef LIBMESH_USE_COMPLEX_NUMBERS
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif

      if (im != .0)
        {
          ierr = PetscPrintf(this->comm().get()," %9f%+9f i %12f\n", re, im, error);
          LIBMESH_CHKERR(ierr);
        }
      else
        {
          ierr = PetscPrintf(this->comm().get(),"   %12f       %12f\n", re, error);
          LIBMESH_CHKERR(ierr);
        }
    }

  ierr = PetscPrintf(this->comm().get(),"\n" );
  LIBMESH_CHKERR(ierr);
#endif // DEBUG

  // return the number of converged eigenpairs
  // and the number of iterations
  return std::make_pair(nconv, its);
}

template <typename T>
void SlepcQuadEigenSolver<T>::set_slepc_solver_type()
{
  PetscErrorCode ierr = 0;

  switch (this->_eigen_solver_type)
    {
    case TOAR:
      ierr = PEPSetType (_pep, (char *) PEPTOAR);    LIBMESH_CHKERR(ierr); return;
    case STOAR:
      ierr = PEPSetType (_pep, (char *) PEPSTOAR); LIBMESH_CHKERR(ierr); return;
    case QARNOLDI:
      ierr = PEPSetType (_pep, (char *) PEPQARNOLDI);   LIBMESH_CHKERR(ierr); return;
    case LINEAR:
      ierr = PEPSetType (_pep, (char *) PEPLINEAR);  LIBMESH_CHKERR(ierr); return;
    case PJD:
      ierr = PEPSetType (_pep, (char *) PEPJD);  LIBMESH_CHKERR(ierr); return;

    default:
      libMesh::err << "ERROR:  Unsupported SLEPc Quadratic Eigen Solver: "
                   << Utility::enum_to_string(this->_eigen_solver_type) << std::endl
                   << "Continuing with SLEPc defaults" << std::endl;
    }
}

template <typename T>
void SlepcQuadEigenSolver<T>:: set_slepc_problem_type()
{
  PetscErrorCode ierr = 0;

  switch (this->_eigen_problem_type)
    {
    case PEPGEN:
      ierr = PEPSetProblemType (_pep, PEP_GENERAL);  LIBMESH_CHKERR(ierr); return;

    default:
      libMesh::err << "ERROR:  Unsupported SLEPc Quadratic Eigen Problem: "
                   << this->_eigen_problem_type << std::endl
                   << "Continuing with SLEPc defaults" << std::endl;
    }
}

template <typename T>
void SlepcQuadEigenSolver<T>:: set_slepc_position_of_spectrum()
{
  PetscErrorCode ierr = 0;

  switch (this->_position_of_spectrum)
    {
    case LARGEST_MAGNITUDE:
      {
        ierr = PEPSetWhichEigenpairs (_pep, PEP_LARGEST_MAGNITUDE);
        LIBMESH_CHKERR(ierr);
        return;
      }
    case SMALLEST_MAGNITUDE:
      {
        ierr = PEPSetWhichEigenpairs (_pep, PEP_SMALLEST_MAGNITUDE);
        LIBMESH_CHKERR(ierr);
        return;
      }
    case LARGEST_REAL:
      {
        ierr = PEPSetWhichEigenpairs (_pep, PEP_LARGEST_REAL);
        LIBMESH_CHKERR(ierr);
        return;
      }
    case SMALLEST_REAL:
      {
        ierr = PEPSetWhichEigenpairs (_pep, PEP_SMALLEST_REAL);
        LIBMESH_CHKERR(ierr);
        return;
      }
    case LARGEST_IMAGINARY:
      {
        ierr = PEPSetWhichEigenpairs (_pep, PEP_LARGEST_IMAGINARY);
        LIBMESH_CHKERR(ierr);
        return;
      }
    case SMALLEST_IMAGINARY:
      {
        ierr = PEPSetWhichEigenpairs (_pep, PEP_SMALLEST_IMAGINARY);
        LIBMESH_CHKERR(ierr);
        return;
      }

      // The PEP_TARGET_XXX enums were added in SLEPc 3.1
#if !SLEPC_VERSION_LESS_THAN(3,1,0)
    case TARGET_MAGNITUDE:
      {
        ierr = PEPSetTarget(_pep, this->_target_val);
        LIBMESH_CHKERR(ierr);
        ierr = PEPSetWhichEigenpairs (_pep, PEP_TARGET_MAGNITUDE);
        LIBMESH_CHKERR(ierr);
        return;
      }
    case TARGET_REAL:
      {
        ierr = PEPSetTarget(_pep, this->_target_val);
        LIBMESH_CHKERR(ierr);
        ierr = PEPSetWhichEigenpairs (_pep,PEP_TARGET_REAL);
        LIBMESH_CHKERR(ierr);
        return;
      }
    case TARGET_IMAGINARY:
      {
        ierr = PEPSetTarget(_pep, this->_target_val);
        LIBMESH_CHKERR(ierr);
        ierr = PEPSetWhichEigenpairs (_pep, PEP_TARGET_IMAGINARY);
        LIBMESH_CHKERR(ierr);
        return;
      }
#endif

    default:
      libmesh_error_msg("ERROR:  Unsupported SLEPc position of spectrum: " << this->_position_of_spectrum);
    }
}

template <typename T>
std::pair<Real, Real> SlepcQuadEigenSolver<T>::get_eigenpair(dof_id_type i,
                                                         NumericVector<T> & solution_in)
{
  PetscErrorCode ierr=0;

  PetscReal re, im;

  // Make sure the NumericVector passed in is really a PetscVector
  PetscVector<T> * solution = dynamic_cast<PetscVector<T> *>(&solution_in);

  if (!solution)
    libmesh_error_msg("Error getting eigenvector: input vector must be a PetscVector.");

  // real and imaginary part of the ith eigenvalue.
  PetscScalar kr, ki;

  solution->close();

  ierr = PEPGetEigenpair(_pep, i, &kr, &ki, solution->vec(), PETSC_NULL);
  LIBMESH_CHKERR(ierr);

#ifdef LIBMESH_USE_COMPLEX_NUMBERS
  re = PetscRealPart(kr);
  im = PetscImaginaryPart(kr);
#else
  re = kr;
  im = ki;
#endif

  return std::make_pair(re, im);
}

template <typename T>
std::pair<Real, Real> SlepcQuadEigenSolver<T>::get_eigenvalue(dof_id_type i)
{
  PetscErrorCode ierr=0;

  PetscReal re, im;

  // real and imaginary part of the ith eigenvalue.
  PetscScalar kr, ki;

  ierr = PEPGetEigenpair(_pep, i, &kr, &ki, PETSC_NULL, PETSC_NULL);
  LIBMESH_CHKERR(ierr);

#ifdef LIBMESH_USE_COMPLEX_NUMBERS
  re = PetscRealPart(kr);
  im = PetscImaginaryPart(kr);
#else
  re = kr;
  im = ki;
#endif

  return std::make_pair(re, im);
}

template <typename T>
Real SlepcQuadEigenSolver<T>::get_relative_error(unsigned int i)
{
  PetscErrorCode ierr=0;
  PetscReal error;

#if SLEPC_VERSION_LESS_THAN(3,6,0)
  ierr = PEPComputeRelativeError(_pep, i, &error);
#else
  ierr = PEPComputeError(_pep, i, PEP_ERROR_RELATIVE, &error);
#endif
  LIBMESH_CHKERR(ierr);

  return error;
}


template <typename T>
void SlepcQuadEigenSolver<T>::set_initial_space(NumericVector<T> & initial_space_in)
{
#if SLEPC_VERSION_LESS_THAN(3,1,0)
  libmesh_error_msg("SLEPc 3.1 is required to call EigenSolver::set_initial_space()");
#else
  this->init();

  PetscErrorCode ierr = 0;

  // Make sure the input vector is actually a PetscVector
  PetscVector<T> * initial_space_petsc_vec =
    dynamic_cast<PetscVector<T> *>(&initial_space_in);

  if (!initial_space_petsc_vec)
    libmesh_error_msg("Error attaching initial space: input vector must be a PetscVector.");

  // Get a handle for the underlying Vec.
  Vec initial_vector = initial_space_petsc_vec->vec();

  ierr = PEPSetInitialSpace(_pep, 1, &initial_vector);
  LIBMESH_CHKERR(ierr);
#endif
}

//------------------------------------------------------------------
// Explicit instantiations
template class SlepcQuadEigenSolver<Number>;

} // namespace libMesh



// #endif // #ifdef LIBMESH_HAVE_SLEPC
