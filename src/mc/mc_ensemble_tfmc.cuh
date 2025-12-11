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

#pragma once
#include "utilities/gpu_vector.cuh"
#include "mc_ensemble_canonical.cuh"

class MC_Ensemble_TFMC : public MC_Ensemble_Canonical
{
public:
  MC_Ensemble_TFMC(const char** param, int num_param, int num_steps_mc);
  virtual ~MC_Ensemble_TFMC(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);

private:
  double displacement_max = 0.1; // maximum displacement length (Angstrom)
  GPU_Vector<double> position_trial; // buffer for trial positions
};
