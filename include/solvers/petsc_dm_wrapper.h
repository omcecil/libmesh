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

#ifndef LIBMESH_PETSC_DM_WRAPPER_H
#define LIBMESH_PETSC_DM_WRAPPER_H

#include "libmesh/libmesh_common.h"

#ifdef LIBMESH_HAVE_PETSC

#include <vector>
#include <memory>
#include <unordered_map>
#include <map>

// PETSc includes
#include "libmesh/ignore_warnings.h"
#include <petsc.h>
#include "libmesh/restore_warnings.h"

namespace libMesh
{
  // Forward declarations
  class System;
  class DofObject;

/**
 * This class defines a wrapper around the PETSc DM infrastructure.
 * By coordinating DM data structures with libMesh, we can use libMesh
 * mesh hierarchies for geometric multigrid. Additionally, by setting the
 * DM data, we can additionally (with or without multigrid) define recursive
 * fieldsplits of our variables.
 *
 * \author Paul T. Bauman, Boris Boutkov
 * \date 2018
 */
class PetscDMWrapper
{
public:

  PetscDMWrapper() = default;

  ~PetscDMWrapper();

  //! Destroys and clears all build DM-related data
  void clear();

  void init_and_attach_petscdm(System & system, SNES & snes);

private:

  //! Vector of DMs for all grid levels
  std::vector<std::unique_ptr<DM>> _dms;

  //! Vector of PETScSections for all grid levels
  std::vector<std::unique_ptr<PetscSection>> _sections;

  //! Vector of star forests for all grid levels
  std::vector<std::unique_ptr<PetscSF>> _star_forests;

  //! Init all the n_mesh_level dependent data structures
  void init_dm_data(unsigned int n_levels);

  //! Get reference to DM for the given mesh level
  /**
   * init_dm_data() should be called before this function.
   */
  DM & get_dm(unsigned int level)
  { libmesh_assert(level < _dms.size());
    return *(_dms[level].get()); }

  //! Get reference to PetscSection for the given mesh level
  /**
   * init_dm_data() should be called before this function.
   */
  PetscSection & get_section(unsigned int level)
  { libmesh_assert(level < _sections.size());
    return *(_sections[level].get()); }

  //! Get reference to PetscSF for the given mesh level
  /**
   * init_dm_data() should be called before this function.
   */
  PetscSF & get_star_forest(unsigned int level)
  { libmesh_assert(level < _star_forests.size());
    return *(_star_forests[level].get()); }

  //! Takes System, empty PetscSection and populates the PetscSection
  /**
   * Take the System in its current state and an empty PetscSection and then
   * populate the PetscSection. The PetscSection is comprised of global "point"
   * numbers, where a "point" in PetscDM parlance is a geometric entity: node, edge,
   * face, or element. Then, we also add the DoF numbering for each variable
   * for each of the "points". The PetscSection, together the with PetscSF
   * will allow for recursive fieldsplits from the command line using PETSc.
   */
  void build_section(const System & system, PetscSection & section);

  //! Takes System, empty PetscSF and populates the PetscSF
  /**
   * The PetscSF (star forest) is a cousin of PetscSection. PetscSection
   * has the DoF info, and PetscSF gives the parallel distribution of the
   * DoF info. So PetscSF should only be necessary when we have more than
   * one MPI rank. Essentially, we are copying the DofMap.send_list(): we
   * are specifying the local dofs, what rank communicates that dof info
   * (for off-processor dofs that are communicated) and the dofs local
   * index on that rank.
   *
   * https://jedbrown.org/files/StarForest.pdf
   */
  void build_sf( const System & system, PetscSF & star_forest );

  //! Helper function for build_section.
  /**
   * This function will count how many "points" on the current processor have
   * DoFs associated with them and give that count to PETSc. We need to cache
   * a mapping between the global node id and our local count that we do in this
   * function because we will need the local number again in the add_dofs_to_section
   * function.
   */
  void set_point_range_in_section( const System & system,
                                   PetscSection & section,
                                   std::unordered_map<dof_id_type,dof_id_type> & node_map,
                                   std::unordered_map<dof_id_type,dof_id_type> & elem_map,
                                   std::map<dof_id_type,unsigned int> & scalar_map);

  //! Helper function for build_section.
  /**
   * This function will set the DoF info for each "point" in the PetscSection.
   */
  void add_dofs_to_section (const System & system,
                            PetscSection & section,
                            const std::unordered_map<dof_id_type,dof_id_type> & node_map,
                            const std::unordered_map<dof_id_type,dof_id_type> & elem_map,
                            const std::map<dof_id_type,unsigned int> & scalar_map);

  //! Helper function to sanity check PetscSection construction
  /**
   * The PetscSection contains local dof information. This helper function just facilitates
   * sanity checking that in fact it only has n_local_dofs.
   */
  dof_id_type check_section_n_dofs( const System & system, PetscSection & section );

  //! Helper function to reduce code duplication when setting dofs in section
  void add_dofs_helper (const System & system,
                        const DofObject & dof_object,
                        dof_id_type local_id,
                        PetscSection & section);

};

}

#endif // #ifdef LIBMESH_HAVE_PETSC

#endif // LIBMESH_PETSC_DM_WRAPPER_H
