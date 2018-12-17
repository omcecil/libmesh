// The libMesh Finite Element Library.
// Copyright (C) 2002-2018 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

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


#ifndef LIBMESH_QUADRATURE_CONICAL_H
#define LIBMESH_QUADRATURE_CONICAL_H

// Local includes
#include "libmesh/quadrature.h"

namespace libMesh
{

/**
 * This class implements the so-called conical product quadrature
 * rules for Tri and Tet elements. These rules are generally
 * non-optimal in the number of evaluation points, but have the following nice
 * properties:
 * .) All positive weights.
 * .) Easily constructed for any order given the underlying 1D Gauss
 *    and Jacobi quadrature rules.
 * The construction of these rules is given by e.g.
 * A. H. Stroud, "Approximate Calculation of Multiple Integrals.", 1972.
 *
 * \author John W. Peterson
 * \date 2008
 * \brief Conical product quadrature rules for Tri and Tet elements.
 */
class QConical final : public QBase
{
public:

  /**
   * Constructor.  Declares the order of the quadrature rule.
   */
  QConical (unsigned int d,
            Order o=INVALID_ORDER) :
    QBase(d,o)
  {}

  /**
   * Copy/move ctor, copy/move assignment operator, and destructor are
   * all explicitly defaulted for this simple class.
   */
  QConical (const QConical &) = default;
  QConical (QConical &&) = default;
  QConical & operator= (const QConical &) = default;
  QConical & operator= (QConical &&) = default;
  virtual ~QConical() = default;

  /**
   * \returns The QuadratureType for this class.
   */
  virtual QuadratureType type() const override;

private:

  /**
   * The optimal "conical product" rule in 1D is simply Gauss.
   */
  virtual void init_1D (const ElemType,
                        unsigned int = 0) override;

  /**
   * The conical product rules are defined in 2D only for Tris.
   */
  virtual void init_2D (const ElemType _type=INVALID_ELEM,
                        unsigned int p_level=0) override;
  /**
   * The conical product rules are defined in 3D only for Tets.
   */
  virtual void init_3D (const ElemType _type=INVALID_ELEM,
                        unsigned int p_level=0) override;

  /**
   * Implementation of conical product rule for a Tri in 2D of
   * order = _order+2*p.
   */
  void conical_product_tri(unsigned int p);

  /**
   * Implementation of conical product rule for a Tet in 3D of
   * order = _order+2*p.
   */
  void conical_product_tet(unsigned int p);

  /**
   * Implementation of conical product rule for a Pyramid in 3D of
   * order = _order+2*p.
   */
  void conical_product_pyramid(unsigned int p);
};

} // namespace libMesh

#endif // LIBMESH_QUADRATURE_CONICAL_H
