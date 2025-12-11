/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
A TFMC ensemble for MCMD. Currently uses the same swap-based moves as the
canonical ensemble but keeps the implementation separate for future TFMC-specific
extensions.
------------------------------------------------------------------------------*/

#include "mc_ensemble_tfmc.cuh"

MC_Ensemble_TFMC::MC_Ensemble_TFMC(const char** param, int num_param, int num_steps_mc_input)
  : MC_Ensemble_Canonical(param, num_param, num_steps_mc_input)
{}

MC_Ensemble_TFMC::~MC_Ensemble_TFMC(void) {}

void MC_Ensemble_TFMC::compute(
  int md_step,
  double temperature,
  Atom& atom,
  Box& box,
  std::vector<Group>& groups,
  int grouping_method,
  int group_id)
{
  // For now TFMC reuses the canonical swap move set.
  MC_Ensemble_Canonical::compute(md_step, temperature, atom, box, groups, grouping_method, group_id);
}
