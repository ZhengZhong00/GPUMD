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
Force-biased translational Monte Carlo (TFMC) ensemble for MCMD. This follows the
LAMMPS tfMC strategy: trial displacements are generated with a force-dependent
distribution and accepted/rejected with a Metropolis criterion using local NEP
energy evaluations.
------------------------------------------------------------------------------*/

#include "mc_ensemble_tfmc.cuh"
#include "utilities/common.cuh"
#include <algorithm>
#include <cmath>

namespace
{
static __global__ void displace_atom(
  const int atom_index,
  const double dx,
  const double dy,
  const double dz,
  double* g_x,
  double* g_y,
  double* g_z)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_x[atom_index] += dx;
    g_y[atom_index] += dy;
    g_z[atom_index] += dz;
  }
}
} // namespace

MC_Ensemble_TFMC::MC_Ensemble_TFMC(const char** param, int num_param, int num_steps_mc_input)
  : MC_Ensemble_Canonical(param, num_param, num_steps_mc_input)
{
  // optional 7th parameter sets the maximum displacement length
  if (num_param >= 7) {
    double dmax_input = 0.0;
    if (!is_valid_real(param[6], &dmax_input) || dmax_input <= 0.0) {
      PRINT_INPUT_ERROR("tfmc displacement must be positive.\n");
    }
    displacement_max = dmax_input;
  }
}

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
  if (check_if_small_box(nep_energy.paramb.rc_radial, box)) {
    printf("Cannot use small box for MCMD.\n");
    exit(1);
  }

  if (type_before.size() < atom.number_of_atoms) {
    type_before.resize(atom.number_of_atoms);
    type_after.resize(atom.number_of_atoms);
  }

  if (position_trial.size() < atom.number_of_atoms * 3) {
    position_trial.resize(atom.number_of_atoms * 3);
  }

  // Host buffers for force and position needed to build biased moves.
  std::vector<double> force_cpu(atom.number_of_atoms * 3, 0.0);
  atom.force_per_atom.copy_to_host(force_cpu.data());

  // compute mass minimum once
  double mass_min = *std::min_element(atom.cpu_mass.begin(), atom.cpu_mass.end());

  int group_size =
    grouping_method >= 0 ? groups[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  std::uniform_int_distribution<int> r1(0, group_size - 1);
  std::uniform_real_distribution<double> r_uni(0.0, 1.0);

  int num_accepted = 0;
  for (int step = 0; step < num_steps_mc; ++step) {

    int i = grouping_method >= 0
              ? groups[grouping_method]
                  .cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + r1(rng)]
              : r1(rng);

    // build biased displacement following LAMMPS tfmc approach
    double dx = 0.0, dy = 0.0, dz = 0.0;
    double mass_i = atom.cpu_mass[i];
    double d_i = displacement_max * pow(mass_min / mass_i, 0.25);
    double forces[3] = {force_cpu[i], force_cpu[i + atom.number_of_atoms], force_cpu[i + 2 * atom.number_of_atoms]};
    for (int dim = 0; dim < 3; ++dim) {
      double gamma = forces[dim] * d_i / (2.0 * K_B * temperature);
      double gamma_exp = exp(gamma);
      double gamma_expi = 1.0 / gamma_exp;
      double p_acc = 0.0;
      double p_ran = 1.0;
      double xi = 0.0;
      while (p_acc < p_ran) {
        xi = 2.0 * r_uni(rng) - 1.0;
        p_ran = r_uni(rng);
        if (xi < 0) {
          p_acc = exp(2.0 * xi * gamma) * gamma_exp - gamma_expi;
          p_acc = p_acc / (gamma_exp - gamma_expi);
        } else if (xi > 0) {
          p_acc = gamma_exp - exp(2.0 * xi * gamma) * gamma_expi;
          p_acc = p_acc / (gamma_exp - gamma_expi);
        } else {
          p_acc = 1.0;
        }
      }
      if (dim == 0) dx = xi * d_i;
      if (dim == 1) dy = xi * d_i;
      if (dim == 2) dz = xi * d_i;
    }

    // Prepare type buffers (unchanged for TFMC)
    get_types<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      i,
      i,
      atom.cpu_type[i],
      atom.cpu_type[i],
      atom.type.data(),
      type_before.data(),
      type_after.data());
    GPU_CHECK_KERNEL

    // neighbor list around atom i (using original coordinates)
    CHECK(gpuMemset(NN_ij.data(), 0, sizeof(int)));
    get_neighbors_of_i_and_j<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      i,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_ij.data(),
      NL_ij.data());
    GPU_CHECK_KERNEL

    // create a trial coordinate buffer on device
    size_t bytes = sizeof(double) * atom.number_of_atoms;
    CHECK(gpuMemcpy(position_trial.data(), atom.position_per_atom.data(), bytes, cudaMemcpyDeviceToDevice));
    CHECK(gpuMemcpy(
      position_trial.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms,
      bytes,
      cudaMemcpyDeviceToDevice));
    CHECK(gpuMemcpy(
      position_trial.data() + 2 * atom.number_of_atoms,
      atom.position_per_atom.data() + 2 * atom.number_of_atoms,
      bytes,
      cudaMemcpyDeviceToDevice));

    displace_atom<<<1, 1>>>(
      i,
      dx,
      dy,
      dz,
      position_trial.data(),
      position_trial.data() + atom.number_of_atoms,
      position_trial.data() + 2 * atom.number_of_atoms);
    GPU_CHECK_KERNEL

    get_neighbors_of_i_and_j<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      i,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      position_trial.data(),
      position_trial.data() + atom.number_of_atoms,
      position_trial.data() + atom.number_of_atoms * 2,
      NN_ij.data(),
      NL_ij.data());
    GPU_CHECK_KERNEL

    int NN_i_cpu = 0;
    NN_ij.copy_to_host(&NN_i_cpu);

    find_local_types<<<(NN_i_cpu - 1) / 64 + 1, 64>>>(
      NN_i_cpu,
      NL_ij.data(),
      type_before.data(),
      type_after.data(),
      local_type_before.data(),
      local_type_after.data());
    GPU_CHECK_KERNEL

    CHECK(gpuMemset(NN_radial.data(), 0, sizeof(int) * NN_radial.size()));
    CHECK(gpuMemset(NN_angular.data(), 0, sizeof(int) * NN_angular.size()));

    create_inputs_for_energy_calculator<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      NN_i_cpu,
      NL_ij.data(),
      box,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      nep_energy.paramb.rc_angular * nep_energy.paramb.rc_angular,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      type_before.data(),
      type_after.data(),
      NN_radial.data(),
      NN_angular.data(),
      t2_radial_before.data(),
      t2_radial_after.data(),
      t2_angular_before.data(),
      t2_angular_after.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data());
    GPU_CHECK_KERNEL

    nep_energy.find_energy(
      NN_i_cpu,
      NN_radial.data(),
      NN_angular.data(),
      local_type_before.data(),
      t2_radial_before.data(),
      t2_angular_before.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      pe_before.data());

    CHECK(gpuMemset(NN_radial.data(), 0, sizeof(int) * NN_radial.size()));
    CHECK(gpuMemset(NN_angular.data(), 0, sizeof(int) * NN_angular.size()));

    // reuse buffers but feed trial coordinates for "after"
    create_inputs_for_energy_calculator<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      NN_i_cpu,
      NL_ij.data(),
      box,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      nep_energy.paramb.rc_angular * nep_energy.paramb.rc_angular,
      position_trial.data(),
      position_trial.data() + atom.number_of_atoms,
      position_trial.data() + atom.number_of_atoms * 2,
      type_before.data(),
      type_after.data(),
      NN_radial.data(),
      NN_angular.data(),
      t2_radial_after.data(),
      t2_radial_after.data(),
      t2_angular_after.data(),
      t2_angular_after.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data());
    GPU_CHECK_KERNEL

    nep_energy.find_energy(
      NN_i_cpu,
      NN_radial.data(),
      NN_angular.data(),
      local_type_after.data(),
      t2_radial_after.data(),
      t2_angular_after.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      pe_after.data());

    std::vector<float> pe_before_cpu(NN_i_cpu);
    std::vector<float> pe_after_cpu(NN_i_cpu);
    pe_before.copy_to_host(pe_before_cpu.data(), NN_i_cpu);
    pe_after.copy_to_host(pe_after_cpu.data(), NN_i_cpu);
    float pe_before_total = 0.0f;
    float pe_after_total = 0.0f;
    for (int n = 0; n < NN_i_cpu; ++n) {
      pe_before_total += pe_before_cpu[n];
      pe_after_total += pe_after_cpu[n];
    }

    float energy_difference = pe_after_total - pe_before_total;
    double probability = exp(-energy_difference / (K_B * temperature));
    double random_number = r_uni(rng);

    if (random_number < probability) {
      ++num_accepted;

      // apply displacement to device position
      displace_atom<<<1, 1>>>(
        i,
        dx,
        dy,
        dz,
        atom.position_per_atom.data(),
        atom.position_per_atom.data() + atom.number_of_atoms,
        atom.position_per_atom.data() + atom.number_of_atoms * 2);
      GPU_CHECK_KERNEL

      atom.cpu_position_per_atom[i] += dx;
      atom.cpu_position_per_atom[i + atom.number_of_atoms] += dy;
      atom.cpu_position_per_atom[i + 2 * atom.number_of_atoms] += dz;
    }
  }

  mc_output << md_step << "  " << num_accepted / double(num_steps_mc) << std::endl;
}
