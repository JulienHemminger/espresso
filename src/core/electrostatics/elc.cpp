/*
 * Copyright (C) 2010-2026 The ESPResSo project
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
 *   Max-Planck-Institute for Polymer Research, Theory Group
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <config/config.hpp>

#ifdef ESPRESSO_P3M

#include "electrostatics/elc.hpp"

#include "electrostatics/coulomb.hpp"
#include "electrostatics/p3m.hpp"

#include "BoxGeometry.hpp"
#include "Particle.hpp"
#include "ParticlePropertyIterator.hpp"
#include "ParticleRange.hpp"
#include "cell_system/CellStructure.hpp"
#include "communication.hpp"
#include "errorhandling.hpp"
#include "system/System.hpp"

#include <utils/math/sqr.hpp>

#include <Kokkos_Core.hpp>

#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/range/combine.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>
#include <iostream>   // added for print statements
#include <iomanip>    // added for std::setprecision

/** \name Product decomposition data organization
 *  For the cell blocks it is assumed that the lower blocks part is in the
 *  lower half. This has to have positive sign, so that has to be first.
 */
/**@{*/
#define POQESP 0
#define POQECP 1
#define POQESM 2
#define POQECM 3

#define PQESSP 0
#define PQESCP 1
#define PQECSP 2
#define PQECCP 3
#define PQESSM 4
#define PQESCM 5
#define PQECSM 6
#define PQECCM 7
/**@}*/

/** ELC axes (x and y directions)*/
enum class PoQ : int { P, Q };
/** ELC charge sum/assign protocol: real charges, image charges, or both. */
enum class ChargeProtocol : int { REAL, IMAGE, BOTH };

/** temporary buffers for product decomposition */
static std::vector<double> partblk;
/** collected data from the other cells */
static double gblcblk[8];

/** structure for caching sin and cos values */
struct SCCache {
  double s, c;
};

/** Cached sin/cos values along the x-axis and y-axis */
/**@{*/
static std::vector<SCCache> scxcache;
static std::vector<SCCache> scycache;
/**@}*/

/**
 * @brief Calculate cached sin/cos values for one direction.
 *
 * @tparam dir Index of the dimension to consider (e.g. 0 for x ...).
 *
 * @param particles Particle to calculate values for
 * @param n_freq Number of frequencies to calculate per particle
 * @param u Inverse box length
 * @return Calculated values.
 */
template <std::size_t dir>
static std::vector<SCCache> calc_sc_cache(ParticleRange const &particles,
                                          std::size_t n_freq, double u) {
  auto constexpr c_2pi = 2. * std::numbers::pi;
  auto const n_part = particles.size();
  std::vector<SCCache> ret(n_freq * n_part);

  for (std::size_t freq = 1; freq <= n_freq; freq++) {
    auto const pref = c_2pi * u * static_cast<double>(freq);

    std::size_t o = (freq - 1) * n_part;
    for (auto const &p : particles) {
      auto const arg = pref * p.pos()[dir];
      ret[o++] = {sin(arg), cos(arg)};
    }
  }

  return ret;
}

static std::pair<std::size_t, std::size_t>
prepare_sc_cache(ParticleRange const &particles, BoxGeometry const &box_geo,
                 double far_cut) {
  // TODO possibly unused function
  assert(far_cut >= 0.);
  auto const n_freq_x =
      static_cast<std::size_t>(std::ceil(far_cut * box_geo.length()[0]) + 1.);
  auto const n_freq_y =
      static_cast<std::size_t>(std::ceil(far_cut * box_geo.length()[1]) + 1.);
  auto const u_x = box_geo.length_inv()[0];
  auto const u_y = box_geo.length_inv()[1];
  scxcache = calc_sc_cache<0>(particles, n_freq_x, u_x);
  scycache = calc_sc_cache<1>(particles, n_freq_y, u_y);
  return {n_freq_x, n_freq_y};
}

/*****************************************************************/
/* data distribution */
/*****************************************************************/

static void clear_vec(double *pdc, std::size_t size) {
  for (std::size_t i = 0; i < size; i++)
    pdc[i] = 0.;
}

static void copy_vec(double *pdc_d, double const *pdc_s, std::size_t size) {
  for (std::size_t i = 0; i < size; i++)
    pdc_d[i] = pdc_s[i];
}

static void add_vec(double *pdc_d, double const *pdc_s1, double const *pdc_s2,
                    std::size_t size) {
  for (std::size_t i = 0; i < size; i++)
    pdc_d[i] = pdc_s1[i] + pdc_s2[i];
}

static void addscale_vec(double *pdc_d, double scale, double const *pdc_s1,
                         double const *pdc_s2, std::size_t size) {
  for (std::size_t i = 0; i < size; i++)
    pdc_d[i] = scale * pdc_s1[i] + pdc_s2[i];
}

static void scale_vec(double scale, double *pdc, std::size_t size) {
  for (std::size_t i = 0; i < size; i++)
    pdc[i] *= scale;
}

static double *block(double *p, std::size_t index, std::size_t size) {
  return &p[index * size];
}

static void distribute(std::size_t size) {
  assert(size <= 8);
  double send_buf[8];
  copy_vec(send_buf, gblcblk, size);
  boost::mpi::all_reduce(comm_cart, send_buf, static_cast<int>(size), gblcblk,
                         std::plus<>());
}

void ElectrostaticLayerCorrection::check_gap(Particle const &p) const {
  if (p.q() != 0.) {
    auto const z = p.pos()[2];
    if (z < 0. or z > elc.box_h) {
      runtimeErrorMsg() << "Particle " << p.id() << " entered ELC gap "
                        << "region by " << ((z < 0.) ? z : z - elc.box_h);
    }
  }
}

/*****************************************************************/
/* dipole terms */
/*****************************************************************/

/** Calculate the dipole force.
 *  See @cite yeh99a.
 */
void ElectrostaticLayerCorrection::add_dipole_force() const {
  constexpr std::size_t size = 3;
  auto const &system = get_system();
  auto const &box_geo = *system.box_geo;
  auto const particles = system.cell_structure->local_particles();
  auto const pref = prefactor * 4. * std::numbers::pi / box_geo.volume();

  /* for non-neutral systems, this shift gives the background contribution
   * (rsp. for this shift, the DM of the background is zero) */
  auto const shift = box_geo.length_half()[2];

  // collect moments

  gblcblk[0] = 0.; // sum q_i (z_i - L/2)
  gblcblk[1] = 0.; // sum q_i z_i
  gblcblk[2] = 0.; // sum q_i

  for (auto const &p : particles) {
    check_gap(p);
    auto const q = p.q();
    auto const z = p.pos()[2];

    gblcblk[0] += q * (z - shift);
    gblcblk[1] += q * z;
    gblcblk[2] += q;

    if (elc.dielectric_contrast_on) {
      if (z < elc.space_layer) {
        gblcblk[0] += elc.delta_mid_bot * q * (-z - shift);
        gblcblk[2] += elc.delta_mid_bot * q;
      }
      if (z > (elc.box_h - elc.space_layer)) {
        gblcblk[0] += elc.delta_mid_top * q * (2. * elc.box_h - z - shift);
        gblcblk[2] += elc.delta_mid_top * q;
      }
    }
  }

  gblcblk[0] *= pref;
  gblcblk[1] *= pref / elc.box_h * box_geo.length()[2];
  gblcblk[2] *= pref;

  distribute(size);

  // Yeh + Berkowitz dipole term @cite yeh99a
  auto field_tot = gblcblk[0];

  // Constant potential contribution
  if (elc.const_pot) {
    auto const field_induced = gblcblk[1];
    auto const field_applied = elc.pot_diff / elc.box_h;
    field_tot -= field_applied + field_induced;

    // Print constant potential contributions
    std::cout << std::setprecision(15)
              << "[ELC] Dipole force: field_induced = " << field_induced
              << ", field_applied = " << field_applied
              << ", const_pot correction = " << -(field_applied + field_induced)
              << std::endl;
  }

  // Print the total dipole field before applying to particles
  std::cout << std::setprecision(15)
            << "[ELC] Dipole force: gblcblk[0] (sum q*(z-L/2), scaled) = " << gblcblk[0]
            << ", gblcblk[1] (sum q*z, scaled) = " << gblcblk[1]
            << ", gblcblk[2] (sum q, scaled) = " << gblcblk[2]
            << ", field_tot = " << field_tot
            << std::endl;

  for (auto &p : particles) {
    // Save force before dipole contribution
    auto fz_before = p.force()[2];

    p.force()[2] -= field_tot * p.q();

    double neutralize_contrib = 0.;
    if (!elc.neutralize) {
      // SUBTRACT the forces of the P3M homogeneous neutralizing background
      neutralize_contrib = gblcblk[2] * p.q() * (p.pos()[2] - shift);
      p.force()[2] += neutralize_contrib;
    }

    // Print per-particle dipole force contributions
    std::cout << std::setprecision(15)
              << "[ELC] Dipole force on particle id=" << p.id()
              << ": q=" << p.q()
              << ", z=" << p.pos()[2]
              << ", dipole_Fz_contrib=" << -(field_tot * p.q())
              << ", neutralize_Fz_contrib=" << neutralize_contrib
              << ", total_Fz_after_dipole=" << p.force()[2]
              << std::endl;
  }
}

/** Calculate the dipole energy.
 *  See @cite yeh99a.
 */
double ElectrostaticLayerCorrection::dipole_energy() const {
  constexpr std::size_t size = 7;
  auto const &system = get_system();
  auto const &box_geo = *system.box_geo;
  auto const particles = system.cell_structure->local_particles();
  auto const pref = prefactor * 2. * std::numbers::pi / box_geo.volume();
  auto const lz = box_geo.length()[2];
  /* for nonneutral systems, this shift gives the background contribution
     (rsp. for this shift, the DM of the background is zero) */
  auto const shift = box_geo.length_half()[2];

  // collect moments

  gblcblk[0] = 0.; // sum q_i               primary box
  gblcblk[1] = 0.; // sum q_i               boundary layers
  gblcblk[2] = 0.; // sum q_i (z_i - L/2)   primary box
  gblcblk[3] = 0.; // sum q_i (z_i - L/2)   boundary layers
  gblcblk[4] = 0.; // sum q_i (z_i - L/2)^2 primary box
  gblcblk[5] = 0.; // sum q_i (z_i - L/2)^2 boundary layers
  gblcblk[6] = 0.; // sum q_i z_i           primary box

  for (auto const &p : particles) {
    check_gap(p);
    auto const q = p.q();
    auto const z = p.pos()[2];

    gblcblk[0] += q;
    gblcblk[2] += q * (z - shift);
    gblcblk[4] += q * (Utils::sqr(z - shift));
    gblcblk[6] += q * z;

    if (elc.dielectric_contrast_on) {
      if (z < elc.space_layer) {
        gblcblk[1] += elc.delta_mid_bot * q;
        gblcblk[3] += elc.delta_mid_bot * q * (-z - shift);
        gblcblk[5] += elc.delta_mid_bot * q * (Utils::sqr(-z - shift));
      }
      if (z > (elc.box_h - elc.space_layer)) {
        gblcblk[1] += elc.delta_mid_top * q;
        gblcblk[3] += elc.delta_mid_top * q * (2. * elc.box_h - z - shift);
        gblcblk[5] +=
            elc.delta_mid_top * q * (Utils::sqr(2. * elc.box_h - z - shift));
      }
    }
  }

  distribute(size);

  // Print the collected moments
  std::cout << std::setprecision(15)
            << "[ELC] Dipole energy moments:"
            << " gblcblk[0]= " << gblcblk[0] << " (sum q, primary)"
            << " gblcblk[1]= " << gblcblk[1] << " (sum q, boundary)"
            << " gblcblk[2]= " << gblcblk[2] << " (sum q*(z-L/2), primary)"
            << " gblcblk[3]= " << gblcblk[3] << " (sum q*(z-L/2), boundary)"
            << " gblcblk[4]= " << gblcblk[4] << " (sum q*(z-L/2)^2, primary)"
            << " gblcblk[5]= " << gblcblk[5] << " (sum q*(z-L/2)^2, boundary)"
            << " gblcblk[6]= " << gblcblk[6] << " (sum q*z, primary)"
            << std::endl;

  // Yeh + Berkowitz term @cite yeh99a
  auto energy = 2. * pref * (Utils::sqr(gblcblk[2]) + gblcblk[2] * gblcblk[3]);

  std::cout << std::setprecision(15)
            << "[ELC] Dipole energy: Yeh-Berkowitz term = " << energy
            << std::endl;

  if (!elc.neutralize) {
    // SUBTRACT the energy of the P3M homogeneous neutralizing background
    auto const neutralize_contrib =
        2. * pref *
        (-gblcblk[0] * gblcblk[4] -
         (.25 - .5 / 3.) * Utils::sqr(gblcblk[0] * lz));
    energy += neutralize_contrib;
    std::cout << std::setprecision(15)
              << "[ELC] Dipole energy: neutralizing background subtraction = "
              << neutralize_contrib << std::endl;
  }

  if (elc.dielectric_contrast_on) {
    if (elc.const_pot) {
      // zero potential difference contribution
      auto const zero_pot_contrib = pref / elc.box_h * lz * Utils::sqr(gblcblk[6]);
      energy += zero_pot_contrib;
      std::cout << std::setprecision(15)
                << "[ELC] Dipole energy: const_pot zero-potential-difference contrib = "
                << zero_pot_contrib << std::endl;

      // external potential shift contribution
      auto const ext_pot_contrib = -2. * elc.pot_diff / elc.box_h * gblcblk[6];
      energy += ext_pot_contrib;
      std::cout << std::setprecision(15)
                << "[ELC] Dipole energy: const_pot external potential shift contrib = "
                << ext_pot_contrib << std::endl;
    }

    /* counter the P3M homogeneous background contribution to the
       boundaries. We never need that, since a homogeneous background
       spanning the artificial boundary layers is aphysical. */
    auto const boundary_contrib =
        pref * (-(gblcblk[1] * gblcblk[4] + gblcblk[0] * gblcblk[5]) -
                (1. - 2. / 3.) * gblcblk[0] * gblcblk[1] * Utils::sqr(lz));
    energy += boundary_contrib;
    std::cout << std::setprecision(15)
              << "[ELC] Dipole energy: dielectric boundary P3M background counter = "
              << boundary_contrib << std::endl;
  }

  std::cout << std::setprecision(15)
            << "[ELC] Dipole energy total (node 0 only) = "
            << (this_node == 0 ? energy : 0.) << std::endl;

  return this_node == 0 ? energy : 0.;
}

/*****************************************************************/

struct ImageSum {
  double delta;
  double shift;
  double lz;
  double dci; // delta complement inverse

  ImageSum(double delta, double shift, double lz)
      : delta{delta}, shift{shift}, lz{lz}, dci{1. / (1. - delta)} {}

  /** @brief Image sum from the bottom layer. */
  double b(double q, double z) const {
    return q * dci * (z - 2. * delta * lz * dci) - q * dci * shift;
  }

  /** @brief Image sum from the top layer. */
  double t(double q, double z) const {
    return q * dci * (z + 2. * delta * lz * dci) - q * dci * shift;
  }
};

double ElectrostaticLayerCorrection::z_energy() const {
  constexpr std::size_t size = 4;
  auto const &system = get_system();
  auto const &box_geo = *system.box_geo;
  auto const particles = system.cell_structure->local_particles();
  auto const xy_area_inv = box_geo.length_inv()[0] * box_geo.length_inv()[1];
  auto const pref = prefactor * 2. * std::numbers::pi * xy_area_inv;

  /* for non-neutral systems, this shift gives the background contribution
   * (rsp. for this shift, the DM of the background is zero) */
  auto const shift = box_geo.length_half()[2];
  auto const lz = box_geo.length()[2];

  if (elc.dielectric_contrast_on) {
    if (elc.const_pot) {
      // metallic boundaries
      clear_vec(gblcblk, size);
      for (auto const &p : particles) {
        auto const z = p.pos()[2];
        auto const q = p.q();
        gblcblk[0] += q;
        gblcblk[1] += q * (z - shift);
        if (z < elc.space_layer) {
          gblcblk[2] -= elc.delta_mid_bot * q;
          gblcblk[3] -= elc.delta_mid_bot * q * (-z - shift);
        }
        if (z > (elc.box_h - elc.space_layer)) {
          gblcblk[2] += elc.delta_mid_top * q;
          gblcblk[3] += elc.delta_mid_top * q * (2. * elc.box_h - z - shift);
        }
      }
    } else {
      // dielectric boundaries
      auto const delta = elc.delta_mid_top * elc.delta_mid_bot;
      auto const fac_delta_mid_bot = elc.delta_mid_bot / (1. - delta);
      auto const fac_delta_mid_top = elc.delta_mid_top / (1. - delta);
      auto const fac_delta = delta / (1. - delta);
      clear_vec(gblcblk, size);
      auto const h = elc.box_h;
      ImageSum const image_sum{delta, shift, lz};
      for (auto const &p : particles) {
        auto const z = p.pos()[2];
        auto const q = p.q();
        gblcblk[0] += q;
        gblcblk[1] += q * (z - shift);
        if (elc.dielectric_contrast_on) {
          if (z < elc.space_layer) {
            gblcblk[2] += fac_delta * (elc.delta_mid_bot + 1.) * q;
            gblcblk[3] +=
                q * (image_sum.b(elc.delta_mid_bot * delta, -(2. * h + z)) +
                     image_sum.b(delta, -(2. * h - z)));
          } else {
            gblcblk[2] += fac_delta_mid_bot * (1. + elc.delta_mid_top) * q;
            gblcblk[3] += q * (image_sum.b(elc.delta_mid_bot, -z) +
                               image_sum.b(delta, -(2. * h - z)));
          }
          if (z > (h - elc.space_layer)) {
            // note the minus sign here which is required due to |z_i-z_j|
            gblcblk[2] -= fac_delta * (elc.delta_mid_top + 1.) * q;
            gblcblk[3] -=
                q * (image_sum.t(elc.delta_mid_top * delta, 4. * h - z) +
                     image_sum.t(delta, 2. * h + z));
          } else {
            // note the minus sign here which is required due to |z_i-z_j|
            gblcblk[2] -= fac_delta_mid_top * (1. + elc.delta_mid_bot) * q;
            gblcblk[3] -= q * (image_sum.t(elc.delta_mid_top, 2. * h - z) +
                               image_sum.t(delta, 2. * h + z));
          }
        }
      }
    }
  }
  distribute(size);

  // Print z-energy components
  std::cout << std::setprecision(15)
            << "[ELC] z_energy: gblcblk[0]= " << gblcblk[0]
            << " (sum q), gblcblk[1]= " << gblcblk[1]
            << " (sum q*(z-shift)), gblcblk[2]= " << gblcblk[2]
            << " (image q sum), gblcblk[3]= " << gblcblk[3]
            << " (image q*(z-shift) sum)" << std::endl;

  auto const energy = gblcblk[1] * gblcblk[2] - gblcblk[0] * gblcblk[3];

  std::cout << std::setprecision(15)
            << "[ELC] z_energy raw (gblcblk[1]*gblcblk[2] - gblcblk[0]*gblcblk[3]) = "
            << energy
            << ", z_energy (node 0 only) = "
            << ((this_node == 0) ? -pref * energy : 0.)
            << std::endl;

  return (this_node == 0) ? -pref * energy : 0.;
}

void ElectrostaticLayerCorrection::add_z_force() const {
  constexpr std::size_t size = 1;
  auto const &system = get_system();
  auto const &box_geo = *system.box_geo;
  auto const particles = system.cell_structure->local_particles();
  auto const xy_area_inv = box_geo.length_inv()[0] * box_geo.length_inv()[1];
  auto const pref = prefactor * 2. * std::numbers::pi * xy_area_inv;

  if (elc.dielectric_contrast_on) {
    if (elc.const_pot) {
      // metallic boundaries
      clear_vec(gblcblk, size);
      /* just counter the 2 pi |z| contribution stemming from P3M */
      for (auto const &p : particles) {
        auto const z = p.pos()[2];
        auto const q = p.q();
        if (z < elc.space_layer)
          gblcblk[0] -= elc.delta_mid_bot * q;
        if (z > (elc.box_h - elc.space_layer))
          gblcblk[0] += elc.delta_mid_top * q;
      }
    } else {
      // dielectric boundaries
      auto const delta = elc.delta_mid_top * elc.delta_mid_bot;
      auto const fac_delta_mid_bot = elc.delta_mid_bot / (1. - delta);
      auto const fac_delta_mid_top = elc.delta_mid_top / (1. - delta);
      auto const fac_delta = delta / (1. - delta);
      clear_vec(gblcblk, size);
      for (auto const &p : particles) {
        auto const z = p.pos()[2];
        auto const q = p.q();
        if (z < elc.space_layer) {
          gblcblk[0] += fac_delta * (elc.delta_mid_bot + 1.) * q;
        } else {
          gblcblk[0] += fac_delta_mid_bot * (elc.delta_mid_top + 1.) * q;
        }
        if (z > (elc.box_h - elc.space_layer)) {
          // note the minus sign here which is required due to |z_i-z_j|
          gblcblk[0] -= fac_delta * (elc.delta_mid_top + 1.) * q;
        } else {
          // note the minus sign here which is required due to |z_i-z_j|
          gblcblk[0] -= fac_delta_mid_top * (elc.delta_mid_bot + 1.) * q;
        }
      }
    }

    gblcblk[0] *= pref;

    distribute(size);

    std::cout << std::setprecision(15)
              << "[ELC] add_z_force: gblcblk[0] (scaled image charge sum) = "
              << gblcblk[0] << std::endl;

    for (auto &p : particles) {
      auto const fz_contrib = gblcblk[0] * p.q();
      p.force()[2] += fz_contrib;
      std::cout << std::setprecision(15)
                << "[ELC] add_z_force on particle id=" << p.id()
                << ": q=" << p.q()
                << ", Fz_z_force_contrib=" << fz_contrib
                << ", total Fz after z_force=" << p.force()[2]
                << std::endl;
    }
  }
}

/*****************************************************************/
/* PoQ exp sum */
/*****************************************************************/

/** \name q=0 or p=0 per frequency code */
/**@{*/
template <PoQ axis>
void setup_PoQ(elc_data const &elc, double prefactor, std::size_t index,
               double omega, ParticleRange const &particles,
               BoxGeometry const &box_geo) {
  assert(index >= 1);
  constexpr std::size_t size = 4;
  auto const xy_area_inv = box_geo.length_inv()[0] * box_geo.length_inv()[1];
  auto const pref_di = prefactor * 4. * std::numbers::pi * xy_area_inv;
  auto const pref = -pref_di / expm1(omega * box_geo.length()[2]);
  double lclimgebot[4], lclimgetop[4], lclimge[4];
  double fac_delta_mid_bot = 1., fac_delta_mid_top = 1., fac_delta = 1.;

  if (elc.dielectric_contrast_on) {
    auto const delta = elc.delta_mid_top * elc.delta_mid_bot;
    auto const fac_elc = 1. / (1. - delta * exp(-omega * 2. * elc.box_h));
    fac_delta_mid_bot = elc.delta_mid_bot * fac_elc;
    fac_delta_mid_top = elc.delta_mid_top * fac_elc;
    fac_delta = fac_delta_mid_bot * elc.delta_mid_top;
  }

  clear_vec(lclimge, size);
  clear_vec(gblcblk, size);
  auto const &sc_cache = (axis == PoQ::P) ? scxcache : scycache;

  std::size_t ic = 0;
  auto const o = (index - 1) * particles.size();
  for (auto const &p : particles) {
    auto const z = p.pos()[2];
    auto const q = p.q();
    auto e = exp(omega * z);

    partblk[size * ic + POQESM] = q * sc_cache[o + ic].s / e;
    partblk[size * ic + POQESP] = q * sc_cache[o + ic].s * e;
    partblk[size * ic + POQECM] = q * sc_cache[o + ic].c / e;
    partblk[size * ic + POQECP] = q * sc_cache[o + ic].c * e;

    add_vec(gblcblk, gblcblk, block(partblk.data(), ic, size), size);

    if (elc.dielectric_contrast_on) {
      if (z < elc.space_layer) { // handle the lower case first
        // negative sign is okay here as the image is located at -z

        e = exp(-omega * z);

        auto const scale = q * elc.delta_mid_bot;

        lclimgebot[POQESM] = sc_cache[o + ic].s / e;
        lclimgebot[POQESP] = sc_cache[o + ic].s * e;
        lclimgebot[POQECM] = sc_cache[o + ic].c / e;
        lclimgebot[POQECP] = sc_cache[o + ic].c * e;

        addscale_vec(gblcblk, scale, lclimgebot, gblcblk, size);

        e = (exp(omega * (-z - 2. * elc.box_h)) * elc.delta_mid_bot +
             exp(omega * (+z - 2. * elc.box_h))) *
            fac_delta;
      } else {
        e = (exp(-omega * z) +
             exp(omega * (z - 2. * elc.box_h)) * elc.delta_mid_top) *
            fac_delta_mid_bot;
      }

      lclimge[POQESP] += q * sc_cache[o + ic].s * e;
      lclimge[POQECP] += q * sc_cache[o + ic].c * e;

      if (z > (elc.box_h - elc.space_layer)) { // handle the upper case now
        e = exp(omega * (2. * elc.box_h - z));

        auto const scale = q * elc.delta_mid_top;

        lclimgetop[POQESM] = sc_cache[o + ic].s / e;
        lclimgetop[POQESP] = sc_cache[o + ic].s * e;
        lclimgetop[POQECM] = sc_cache[o + ic].c / e;
        lclimgetop[POQECP] = sc_cache[o + ic].c * e;

        addscale_vec(gblcblk, scale, lclimgetop, gblcblk, size);

        e = (exp(omega * (+z - 4. * elc.box_h)) * elc.delta_mid_top +
             exp(omega * (-z - 2. * elc.box_h))) *
            fac_delta;
      } else {
        e = (exp(omega * (+z - 2. * elc.box_h)) +
             exp(omega * (-z - 2. * elc.box_h)) * elc.delta_mid_bot) *
            fac_delta_mid_top;
      }

      lclimge[POQESM] += q * sc_cache[o + ic].s * e;
      lclimge[POQECM] += q * sc_cache[o + ic].c * e;
    }

    ++ic;
  }

  scale_vec(pref, gblcblk, size);

  if (elc.dielectric_contrast_on) {
    scale_vec(pref_di, lclimge, size);
    add_vec(gblcblk, gblcblk, lclimge, size);
  }

  // Print setup_PoQ results
  std::cout << std::setprecision(15)
            << "[ELC] setup_PoQ<" << (axis == PoQ::P ? "P" : "Q")
            << "> index=" << index << " omega=" << omega
            << ": gblcblk[POQESP]=" << gblcblk[POQESP]
            << " gblcblk[POQECP]=" << gblcblk[POQECP]
            << " gblcblk[POQESM]=" << gblcblk[POQESM]
            << " gblcblk[POQECM]=" << gblcblk[POQECM]
            << std::endl;
}

template <PoQ axis> void add_PoQ_force(ParticleRange const &particles) {
  constexpr auto i = static_cast<int>(axis);
  constexpr std::size_t size = 4;

  std::size_t ic = 0;
  for (auto &p : particles) {
    auto &force = p.force();

    auto const f_lateral =
        partblk[size * ic + POQESM] * gblcblk[POQECP] -
        partblk[size * ic + POQECM] * gblcblk[POQESP] +
        partblk[size * ic + POQESP] * gblcblk[POQECM] -
        partblk[size * ic + POQECP] * gblcblk[POQESM];

    auto const f_z =
        partblk[size * ic + POQECM] * gblcblk[POQECP] +
        partblk[size * ic + POQESM] * gblcblk[POQESP] -
        partblk[size * ic + POQECP] * gblcblk[POQECM] -
        partblk[size * ic + POQESP] * gblcblk[POQESM];

    force[i] += f_lateral;
    force[2] += f_z;

    std::cout << std::setprecision(15)
              << "[ELC] add_PoQ_force<" << (axis == PoQ::P ? "P" : "Q")
              << "> particle id=" << p.id()
              << ": F[" << i << "]_contrib=" << f_lateral
              << ", Fz_contrib=" << f_z
              << ", total F[" << i << "]=" << force[i]
              << ", total Fz=" << force[2]
              << std::endl;

    ++ic;
  }
}

static double PoQ_energy(double omega, std::size_t n_part) {
  constexpr std::size_t size = 4;

  auto energy = 0.;
  for (std::size_t ic = 0; ic < n_part; ic++) {
    energy += partblk[size * ic + POQECM] * gblcblk[POQECP] +
              partblk[size * ic + POQESM] * gblcblk[POQESP] +
              partblk[size * ic + POQECP] * gblcblk[POQECM] +
              partblk[size * ic + POQESP] * gblcblk[POQESM];
  }

  auto const result = energy / omega;
  std::cout << std::setprecision(15)
            << "[ELC] PoQ_energy: omega=" << omega
            << ", raw_sum=" << energy
            << ", energy/omega=" << result
            << std::endl;

  return result;
}
/**@}*/

/*****************************************************************/
/* PQ particle blocks */
/*****************************************************************/

/** \name p,q <> 0 per frequency code */
/**@{*/
static void setup_PQ(elc_data const &elc, double prefactor, std::size_t index_p,
                     std::size_t index_q, double omega,
                     ParticleRange const &particles,
                     BoxGeometry const &box_geo) {
  assert(index_p >= 1);
  assert(index_q >= 1);
  constexpr std::size_t size = 8;
  auto const xy_area_inv = box_geo.length_inv()[0] * box_geo.length_inv()[1];
  auto const pref_di = prefactor * 8. * std::numbers::pi * xy_area_inv;
  auto const pref = -pref_di / expm1(omega * box_geo.length()[2]);
  double lclimgebot[8], lclimgetop[8], lclimge[8];
  double fac_delta_mid_bot = 1., fac_delta_mid_top = 1., fac_delta = 1.;
  if (elc.dielectric_contrast_on) {
    auto const delta = elc.delta_mid_top * elc.delta_mid_bot;
    auto const fac_elc = 1. / (1. - delta * exp(-omega * 2. * elc.box_h));
    fac_delta_mid_bot = elc.delta_mid_bot * fac_elc;
    fac_delta_mid_top = elc.delta_mid_top * fac_elc;
    fac_delta = fac_delta_mid_bot * elc.delta_mid_top;
  }

  clear_vec(lclimge, size);
  clear_vec(gblcblk, size);

  std::size_t ic = 0;
  auto const ox = (index_p - 1) * particles.size();
  auto const oy = (index_q - 1) * particles.size();
  for (auto const &p : particles) {
    auto const z = p.pos()[2];
    auto const q = p.q();
    auto e = exp(omega * z);

    partblk[size * ic + PQESSM] =
        scxcache[ox + ic].s * scycache[oy + ic].s * q / e;
    partblk[size * ic + PQESCM] =
        scxcache[ox + ic].s * scycache[oy + ic].c * q / e;
    partblk[size * ic + PQECSM] =
        scxcache[ox + ic].c * scycache[oy + ic].s * q / e;
    partblk[size * ic + PQECCM] =
        scxcache[ox + ic].c * scycache[oy + ic].c * q / e;

    partblk[size * ic + PQESSP] =
        scxcache[ox + ic].s * scycache[oy + ic].s * q * e;
    partblk[size * ic + PQESCP] =
        scxcache[ox + ic].s * scycache[oy + ic].c * q * e;
    partblk[size * ic + PQECSP] =
        scxcache[ox + ic].c * scycache[oy + ic].s * q * e;
    partblk[size * ic + PQECCP] =
        scxcache[ox + ic].c * scycache[oy + ic].c * q * e;

    add_vec(gblcblk, gblcblk, block(partblk.data(), ic, size), size);

    if (elc.dielectric_contrast_on) {
      if (z < elc.space_layer) { // handle the lower case first
        // change e to take into account the z position of the images

        e = exp(-omega * z);
        auto const scale = q * elc.delta_mid_bot;

        lclimgebot[PQESSM] = scxcache[ox + ic].s * scycache[oy + ic].s / e;
        lclimgebot[PQESCM] = scxcache[ox + ic].s * scycache[oy + ic].c / e;
        lclimgebot[PQECSM] = scxcache[ox + ic].c * scycache[oy + ic].s / e;
        lclimgebot[PQECCM] = scxcache[ox + ic].c * scycache[oy + ic].c / e;

        lclimgebot[PQESSP] = scxcache[ox + ic].s * scycache[oy + ic].s * e;
        lclimgebot[PQESCP] = scxcache[ox + ic].s * scycache[oy + ic].c * e;
        lclimgebot[PQECSP] = scxcache[ox + ic].c * scycache[oy + ic].s * e;
        lclimgebot[PQECCP] = scxcache[ox + ic].c * scycache[oy + ic].c * e;

        addscale_vec(gblcblk, scale, lclimgebot, gblcblk, size);

        e = (exp(omega * (-z - 2. * elc.box_h)) * elc.delta_mid_bot +
             exp(omega * (+z - 2. * elc.box_h))) *
            fac_delta * q;

      } else {

        e = (exp(-omega * z) +
             exp(omega * (z - 2. * elc.box_h)) * elc.delta_mid_top) *
            fac_delta_mid_bot * q;
      }

      lclimge[PQESSP] += scxcache[ox + ic].s * scycache[oy + ic].s * e;
      lclimge[PQESCP] += scxcache[ox + ic].s * scycache[oy + ic].c * e;
      lclimge[PQECSP] += scxcache[ox + ic].c * scycache[oy + ic].s * e;
      lclimge[PQECCP] += scxcache[ox + ic].c * scycache[oy + ic].c * e;

      if (z > (elc.box_h - elc.space_layer)) { // handle the upper case now

        e = exp(omega * (2. * elc.box_h - z));
        auto const scale = q * elc.delta_mid_top;

        lclimgetop[PQESSM] = scxcache[ox + ic].s * scycache[oy + ic].s / e;
        lclimgetop[PQESCM] = scxcache[ox + ic].s * scycache[oy + ic].c / e;
        lclimgetop[PQECSM] = scxcache[ox + ic].c * scycache[oy + ic].s / e;
        lclimgetop[PQECCM] = scxcache[ox + ic].c * scycache[oy + ic].c / e;

        lclimgetop[PQESSP] = scxcache[ox + ic].s * scycache[oy + ic].s * e;
        lclimgetop[PQESCP] = scxcache[ox + ic].s * scycache[oy + ic].c * e;
        lclimgetop[PQECSP] = scxcache[ox + ic].c * scycache[oy + ic].s * e;
        lclimgetop[PQECCP] = scxcache[ox + ic].c * scycache[oy + ic].c * e;

        addscale_vec(gblcblk, scale, lclimgetop, gblcblk, size);

        e = (exp(omega * (+z - 4. * elc.box_h)) * elc.delta_mid_top +
             exp(omega * (-z - 2. * elc.box_h))) *
            fac_delta * q;

      } else {

        e = (exp(omega * (+z - 2. * elc.box_h)) +
             exp(omega * (-z - 2. * elc.box_h)) * elc.delta_mid_bot) *
            fac_delta_mid_top * q;
      }

      lclimge[PQESSM] += scxcache[ox + ic].s * scycache[oy + ic].s * e;
      lclimge[PQESCM] += scxcache[ox + ic].s * scycache[oy + ic].c * e;
      lclimge[PQECSM] += scxcache[ox + ic].c * scycache[oy + ic].s * e;
      lclimge[PQECCM] += scxcache[ox + ic].c * scycache[oy + ic].c * e;
    }

    ic++;
  }

  scale_vec(pref, gblcblk, size);
  if (elc.dielectric_contrast_on) {
    scale_vec(pref_di, lclimge, size);
    add_vec(gblcblk, gblcblk, lclimge, size);
  }

  // Print setup_PQ results
  std::cout << std::setprecision(15)
            << "[ELC] setup_PQ p=" << index_p << " q=" << index_q
            << " omega=" << omega
            << ": gblcblk[PQESSM]=" << gblcblk[PQESSM]
            << " gblcblk[PQESCM]=" << gblcblk[PQESCM]
            << " gblcblk[PQECSM]=" << gblcblk[PQECSM]
            << " gblcblk[PQECCM]=" << gblcblk[PQECCM]
            << " gblcblk[PQESSP]=" << gblcblk[PQESSP]
            << " gblcblk[PQESCP]=" << gblcblk[PQESCP]
            << " gblcblk[PQECSP]=" << gblcblk[PQECSP]
            << " gblcblk[PQECCP]=" << gblcblk[PQECCP]
            << std::endl;
}

static void add_PQ_force(std::size_t index_p, std::size_t index_q, double omega,
                         ParticleRange const &particles,
                         BoxGeometry const &box_geo) {
  auto constexpr c_2pi = 2. * std::numbers::pi;
  auto const pref_x =
      c_2pi * box_geo.length_inv()[0] * static_cast<double>(index_p) / omega;
  auto const pref_y =
      c_2pi * box_geo.length_inv()[1] * static_cast<double>(index_q) / omega;
  constexpr std::size_t size = 8;

  std::size_t ic = 0;
  for (auto &p : particles) {
    auto &force = p.force();

    auto const fx_contrib =
        pref_x * (partblk[size * ic + PQESCM] * gblcblk[PQECCP] +
                  partblk[size * ic + PQESSM] * gblcblk[PQECSP] -
                  partblk[size * ic + PQECCM] * gblcblk[PQESCP] -
                  partblk[size * ic + PQECSM] * gblcblk[PQESSP] +
                  partblk[size * ic + PQESCP] * gblcblk[PQECCM] +
                  partblk[size * ic + PQESSP] * gblcblk[PQECSM] -
                  partblk[size * ic + PQECCP] * gblcblk[PQESCM] -
                  partblk[size * ic + PQECSP] * gblcblk[PQESSM]);

    auto const fy_contrib =
        pref_y * (partblk[size * ic + PQECSM] * gblcblk[PQECCP] +
                  partblk[size * ic + PQESSM] * gblcblk[PQESCP] -
                  partblk[size * ic + PQECCM] * gblcblk[PQECSP] -
                  partblk[size * ic + PQESCM] * gblcblk[PQESSP] +
                  partblk[size * ic + PQECSP] * gblcblk[PQECCM] +
                  partblk[size * ic + PQESSP] * gblcblk[PQESCM] -
                  partblk[size * ic + PQECCP] * gblcblk[PQECSM] -
                  partblk[size * ic + PQESCP] * gblcblk[PQESSM]);

    auto const fz_contrib =
        (partblk[size * ic + PQECCM] * gblcblk[PQECCP] +
         partblk[size * ic + PQECSM] * gblcblk[PQECSP] +
         partblk[size * ic + PQESCM] * gblcblk[PQESCP] +
         partblk[size * ic + PQESSM] * gblcblk[PQESSP] -
         partblk[size * ic + PQECCP] * gblcblk[PQECCM] -
         partblk[size * ic + PQECSP] * gblcblk[PQECSM] -
         partblk[size * ic + PQESCP] * gblcblk[PQESCM] -
         partblk[size * ic + PQESSP] * gblcblk[PQESSM]);

    force[0] += fx_contrib;
    force[1] += fy_contrib;
    force[2] += fz_contrib;

    std::cout << std::setprecision(15)
              << "[ELC] add_PQ_force p=" << index_p << " q=" << index_q
              << " omega=" << omega
              << " particle id=" << p.id()
              << ": Fx_contrib=" << fx_contrib
              << ", Fy_contrib=" << fy_contrib
              << ", Fz_contrib=" << fz_contrib
              << ", total Fx=" << force[0]
              << ", total Fy=" << force[1]
              << ", total Fz=" << force[2]
              << std::endl;

    ic++;
  }
}

static double PQ_energy(double omega, std::size_t n_part) {
  constexpr std::size_t size = 8;

  auto energy = 0.;
  for (std::size_t ic = 0; ic < n_part; ic++) {
    energy += partblk[size * ic + PQECCM] * gblcblk[PQECCP] +
              partblk[size * ic + PQECSM] * gblcblk[PQECSP] +
              partblk[size * ic + PQESCM] * gblcblk[PQESCP] +
              partblk[size * ic + PQESSM] * gblcblk[PQESSP] +
              partblk[size * ic + PQECCP] * gblcblk[PQECCM] +
              partblk[size * ic + PQECSP] * gblcblk[PQECSM] +
              partblk[size * ic + PQESCP] * gblcblk[PQESCM] +
              partblk[size * ic + PQESSP] * gblcblk[PQESSM];
  }

  auto const result = energy / omega;
  std::cout << std::setprecision(15)
            << "[ELC] PQ_energy: omega=" << omega
            << ", raw_sum=" << energy
            << ", energy/omega=" << result
            << std::endl;

  return result;
}
/**@}*/

void ElectrostaticLayerCorrection::add_force() const {
  auto constexpr c_2pi = 2. * std::numbers::pi;
  auto const &system = get_system();
  auto const &box_geo = *system.box_geo;
  auto const particles = system.cell_structure->local_particles();
  auto const n_freqs = prepare_sc_cache(particles, box_geo, elc.far_cut);
  auto const n_scxcache = std::get<0>(n_freqs);
  auto const n_scycache = std::get<1>(n_freqs);
  partblk.resize(particles.size() * 8);

  std::cout << "[ELC] === add_force() begin ===" << std::endl;

  add_dipole_force();
  add_z_force();

  /* the second condition is just for the case of numerical accident */
  for (std::size_t p = 1;
       box_geo.length_inv()[0] * static_cast<double>(p - 1) < elc.far_cut &&
       p <= n_scxcache;
       p++) {
    auto const omega = c_2pi * box_geo.length_inv()[0] * static_cast<double>(p);
    std::cout << std::setprecision(15)
              << "[ELC] add_force: PoQ P loop, p=" << p
              << ", omega=" << omega << std::endl;
    setup_PoQ<PoQ::P>(elc, prefactor, p, omega, particles, box_geo);
    distribute(4);
    add_PoQ_force<PoQ::P>(particles);
  }

  for (std::size_t q = 1;
       box_geo.length_inv()[1] * static_cast<double>(q - 1) < elc.far_cut &&
       q <= n_scycache;
       q++) {
    auto const omega = c_2pi * box_geo.length_inv()[1] * static_cast<double>(q);
    std::cout << std::setprecision(15)
              << "[ELC] add_force: PoQ Q loop, q=" << q
              << ", omega=" << omega << std::endl;
    setup_PoQ<PoQ::Q>(elc, prefactor, q, omega, particles, box_geo);
    distribute(4);
    add_PoQ_force<PoQ::Q>(particles);
  }

  for (std::size_t p = 1;
       box_geo.length_inv()[0] * static_cast<double>(p - 1) < elc.far_cut &&
       p <= n_scxcache;
       p++) {
    for (std::size_t q = 1;
         Utils::sqr(box_geo.length_inv()[0] * static_cast<double>(p - 1)) +
                 Utils::sqr(box_geo.length_inv()[1] *
                            static_cast<double>(q - 1)) <
             elc.far_cut2 &&
         q <= n_scycache;
         q++) {
      auto const omega =
          c_2pi *
          sqrt(Utils::sqr(box_geo.length_inv()[0] * static_cast<double>(p)) +
               Utils::sqr(box_geo.length_inv()[1] * static_cast<double>(q)));
      std::cout << std::setprecision(15)
                << "[ELC] add_force: PQ loop, p=" << p << ", q=" << q
                << ", omega=" << omega << std::endl;
      setup_PQ(elc, prefactor, p, q, omega, particles, box_geo);
      distribute(8);
      add_PQ_force(p, q, omega, particles, box_geo);
    }
  }

  // Print final forces on all particles
  std::cout << "[ELC] === Final forces after add_force() ===" << std::endl;
  for (auto const &p : particles) {
    std::cout << std::setprecision(15)
              << "[ELC] Particle id=" << p.id()
              << ": Fx=" << p.force()[0]
              << ", Fy=" << p.force()[1]
              << ", Fz=" << p.force()[2]
              << std::endl;
  }
  std::cout << "[ELC] === add_force() end ===" << std::endl;
}

double ElectrostaticLayerCorrection::calc_energy() const {

  auto constexpr c_2pi = 2. * std::numbers::pi;
  auto const &system = get_system();
  auto const &box_geo = *system.box_geo;
  auto const particles = system.cell_structure->local_particles();

  std::cout << "[ELC] === calc_energy() begin ===" << std::endl;

  auto const dipole_e = dipole_energy();
  auto const z_e = z_energy();
  auto energy = dipole_e + z_e;

  std::cout << std::setprecision(15)
            << "[ELC] calc_energy: dipole_energy=" << dipole_e
            << ", z_energy=" << z_e
            << ", sum so far=" << energy
            << std::endl;

  auto const n_freqs = prepare_sc_cache(particles, box_geo, elc.far_cut);
  auto const n_scxcache = std::get<0>(n_freqs);
  auto const n_scycache = std::get<1>(n_freqs);

  auto const n_localpart = particles.size();
  partblk.resize(n_localpart * 8);

  /* the second condition is just for the case of numerical accident */
  for (std::size_t p = 1;
       box_geo.length_inv()[0] * static_cast<double>(p - 1) < elc.far_cut &&
       p <= n_scxcache;
       p++) {
    auto const omega = c_2pi * box_geo.length_inv()[0] * static_cast<double>(p);
    setup_PoQ<PoQ::P>(elc, prefactor, p, omega, particles, box_geo);
    distribute(4);
    auto const contrib = PoQ_energy(omega, n_localpart);
    energy += contrib;
    std::cout << std::setprecision(15)
              << "[ELC] calc_energy: PoQ P p=" << p
              << " omega=" << omega
              << " contrib=" << contrib
              << " energy_running=" << energy
              << std::endl;
  }

  for (std::size_t q = 1;
       box_geo.length_inv()[1] * static_cast<double>(q - 1) < elc.far_cut &&
       q <= n_scycache;
       q++) {
    auto const omega = c_2pi * box_geo.length_inv()[1] * static_cast<double>(q);
    setup_PoQ<PoQ::Q>(elc, prefactor, q, omega, particles, box_geo);
    distribute(4);
    auto const contrib = PoQ_energy(omega, n_localpart);
    energy += contrib;
    std::cout << std::setprecision(15)
              << "[ELC] calc_energy: PoQ Q q=" << q
              << " omega=" << omega
              << " contrib=" << contrib
              << " energy_running=" << energy
              << std::endl;
  }

  for (std::size_t p = 1;
       box_geo.length_inv()[0] * static_cast<double>(p - 1) < elc.far_cut &&
       p <= n_scxcache;
       p++) {
    for (std::size_t q = 1;
         Utils::sqr(box_geo.length_inv()[0] * static_cast<double>(p - 1)) +
                 Utils::sqr(box_geo.length_inv()[1] *
                            static_cast<double>(q - 1)) <
             elc.far_cut2 &&
         q <= n_scycache;
         q++) {
      auto const omega =
          c_2pi *
          sqrt(Utils::sqr(box_geo.length_inv()[0] * static_cast<double>(p)) +
               Utils::sqr(box_geo.length_inv()[1] * static_cast<double>(q)));
      setup_PQ(elc, prefactor, p, q, omega, particles, box_geo);
      distribute(8);
      auto const contrib = PQ_energy(omega, n_localpart);
      energy += contrib;
      std::cout << std::setprecision(15)
                << "[ELC] calc_energy: PQ p=" << p << " q=" << q
                << " omega=" << omega
                << " contrib=" << contrib
                << " energy_running=" << energy
                << std::endl;
    }
  }

  /* we count both i<->j and j<->i, so return just half of it */
  auto const final_energy = 0.5 * energy;
  std::cout << std::setprecision(15)
            << "[ELC] calc_energy: total (before *0.5)=" << energy
            << ", final (0.5*total)=" << final_energy
            << std::endl;
  std::cout << "[ELC] === calc_energy() end ===" << std::endl;

  return final_energy;
}

double ElectrostaticLayerCorrection::tune_far_cut() const {
  // Largest reasonable cutoff for far formula
  auto constexpr maximal_far_cut = 50.;
  auto const &box_geo = *get_system().box_geo;
  auto const box_l_x_inv = box_geo.length_inv()[0];
  auto const box_l_y_inv = box_geo.length_inv()[1];
  auto const min_inv_boxl = std::min(box_l_x_inv, box_l_y_inv);
  auto const box_l_z = box_geo.length()[2];
  auto const h = elc.box_h;
  // adjust lz according to dielectric layer method
  auto const lz = (elc.dielectric_contrast_on) ? h + elc.space_layer : box_l_z;

  auto tuned_far_cut = min_inv_boxl;
  double err;
  do {
    // following equation 18 in arnold02d
    auto const pref = 2. * std::numbers::pi * tuned_far_cut;
    auto const sum = pref + 2. * (box_l_x_inv + box_l_y_inv);
    auto const den = expm1(pref * lz);
    auto const num1 = exp(pref * h);
    auto const num2 = 1. / num1; // exp(-pref * h);

    err = 0.5 / den *
          (num1 / (lz - h) * (sum + 1. / (lz - h)) +
           num2 / (lz + h) * (sum + 1. / (lz + h)));

    tuned_far_cut += min_inv_boxl;
  } while (err > elc.maxPWerror and tuned_far_cut < maximal_far_cut);
  if (tuned_far_cut >= maximal_far_cut) {
    throw std::runtime_error("ELC tuning failed: maxPWerror too small");
  }
  return tuned_far_cut - min_inv_boxl;
}

static auto calc_total_charge(CellStructure const &cell_structure) {
  auto local_q = 0.;
  for (auto const &p : cell_structure.local_particles()) {
    local_q += p.q();
  }
  return boost::mpi::all_reduce(comm_cart, local_q, std::plus<>());
}

void ElectrostaticLayerCorrection::sanity_checks_periodicity() const {
  auto const &box_geo = *get_system().box_geo;
  if (!box_geo.periodic(0) || !box_geo.periodic(1) || !box_geo.periodic(2)) {
    throw std::runtime_error("ELC: requires periodicity (True, True, True)");
  }
}

void ElectrostaticLayerCorrection::sanity_checks_dielectric_contrasts() const {
  if (elc.dielectric_contrast_on) {
    auto const &cell_structure = *get_system().cell_structure;
    auto const precision_threshold = std::sqrt(round_error_prec);
    auto const total_charge = std::abs(calc_total_charge(cell_structure));
    if (total_charge >= precision_threshold) {
      if (elc.const_pot) {
        // Disable this line to make ELC work again with non-neutral systems
        // and metallic boundaries
        throw std::runtime_error("ELC does not currently support non-neutral "
                                 "systems with a dielectric contrast.");
      }
      // ELC with non-neutral systems and no fully metallic boundaries
      // does not work
      throw std::runtime_error("ELC does not work for non-neutral systems and "
                               "non-metallic dielectric contrast.");
    }
  }
}

void ElectrostaticLayerCorrection::adapt_solver() {
  // TODO possibly unused function
  std::visit(
      [this](auto &solver) {
        set_prefactor(solver->prefactor);
        solver->adapt_epsilon_elc();
        assert(solver->p3m_params.epsilon == P3M_EPSILON_METALLIC);
      },
      base_solver);
}

void ElectrostaticLayerCorrection::recalc_box_h() {
  m_box_geo = get_system().box_geo.get();
  auto const box_z = m_box_geo->length()[2];
  auto const new_box_h = box_z - elc.gap_size;
  if (new_box_h < 0.) {
    throw std::runtime_error("ELC gap size (" + std::to_string(elc.gap_size) +
                             ") larger than box length in z-direction (" +
                             std::to_string(box_z) + ")");
  }
  elc.box_h = new_box_h;
}

void ElectrostaticLayerCorrection::recalc_space_layer() {
  // TODO possibly unused function
  if (elc.dielectric_contrast_on) {
    auto const p3m_r_cut = std::visit(
        [](auto &solver) { return solver->p3m_params.r_cut; }, base_solver);
    // recalculate the space layer size:
    // 1. set the space_layer to be 1/3 of the gap size, so that box = layer
    elc.space_layer = (1. / 3.) * elc.gap_size;
    // 2. but make sure we don't overlap with the near-field formula
    auto const free_space = elc.gap_size - p3m_r_cut;
    // 3. and make sure the space layer is not bigger than half the actual
    // simulation box, to avoid overlaps
    auto const half_box_h = elc.box_h / 2.;
    auto const max_space_layer = std::min(free_space, half_box_h);
    if (elc.space_layer > max_space_layer) {
      if (max_space_layer <= 0.) {
        throw std::runtime_error("P3M real-space cutoff too large for ELC w/ "
                                 "dielectric contrast");
      }
      elc.space_layer = max_space_layer;
    }
    elc.space_box = elc.gap_size - 2. * elc.space_layer;
  }
}

elc_data::elc_data(double maxPWerror, double gap_size, double far_cut,
                   bool neutralize, double delta_top, double delta_bot,
                   bool with_const_pot, double potential_diff)
    : maxPWerror{maxPWerror}, gap_size{gap_size}, box_h{-1.}, far_cut{far_cut},
      far_cut2{-1.}, far_calculated{far_cut == -1.},
      dielectric_contrast_on{delta_top != 0. or delta_bot != 0.},
      const_pot{with_const_pot and dielectric_contrast_on},
      neutralize{neutralize and !dielectric_contrast_on},
      delta_mid_top{std::clamp(delta_top, -1., +1.)},
      delta_mid_bot{std::clamp(delta_bot, -1., +1.)},
      pot_diff{(with_const_pot) ? potential_diff : 0.},
      // initial setup of parameters, may change later when P3M is finally tuned
      // set the space_layer to be 1/3 of the gap size, so that box = layer
      space_layer{(dielectric_contrast_on) ? gap_size / 3. : 0.},
      space_box{gap_size - ((dielectric_contrast_on) ? 2. * space_layer : 0.)} {
  // TODO possibly unused function
  auto const delta_range = 1. + std::sqrt(round_error_prec);
  if (far_cut <= 0. and not far_calculated) {
    throw std::domain_error("Parameter 'far_cut' must be > 0");
  }
  if (maxPWerror <= 0.) {
    throw std::domain_error("Parameter 'maxPWerror' must be > 0");
  }
  if (gap_size <= 0.) {
    throw std::domain_error("Parameter 'gap_size' must be > 0");
  }
  if (potential_diff != 0. and not with_const_pot) {
    throw std::invalid_argument(
        "Parameter 'const_pot' must be True when 'pot_diff' is non-zero");
  }
  if (delta_top < -delta_range or delta_top > delta_range) {
    throw std::domain_error(
        "Parameter 'delta_mid_top' must be >= -1 and <= +1");
  }
  if (delta_bot < -delta_range or delta_bot > delta_range) {
    throw std::domain_error(
        "Parameter 'delta_mid_bot' must be >= -1 and <= +1");
  }
  /* Dielectric contrasts: the deltas should be either both -1 or both +1 when
   * no constant potential difference is applied. The case of two non-metallic
   * parallel boundaries can only be treated with a constant potential. */
  if (dielectric_contrast_on and not const_pot and
      (std::fabs(1. - delta_mid_top * delta_mid_bot) < round_error_prec)) {
    throw std::domain_error("ELC with two parallel metallic boundaries "
                            "requires the const_pot option");
  }
}

ElectrostaticLayerCorrection::ElectrostaticLayerCorrection(
    elc_data &&parameters, BaseSolver &&solver)
    : elc{parameters}, base_solver{solver} {
  adapt_solver();
}

template <ChargeProtocol protocol, typename combined_ranges>
void charge_assign(elc_data const &elc, CoulombP3M &solver,
                   combined_ranges const &p_q_pos_range) {

  // TODO possibly unused function
  solver.prepare_fft_mesh(protocol == ChargeProtocol::BOTH or
                          protocol == ChargeProtocol::IMAGE);

  // multi-threading -> cache sizes must be equal to the number of particles
  auto constexpr include_neutral_particles = true;

  for (auto zipped : p_q_pos_range) {
    auto const p_q = boost::get<0>(zipped);
    auto const &p_pos = boost::get<1>(zipped);
    if (include_neutral_particles or p_q != 0.) {
      // assign real charges
      if (protocol == ChargeProtocol::BOTH or
          protocol == ChargeProtocol::REAL) {
        solver.assign_charge(p_q, p_pos, false);
      }
      // assign image charges
      if (protocol == ChargeProtocol::BOTH or
          protocol == ChargeProtocol::IMAGE) {
        if (p_pos[2] < elc.space_layer) {
          auto const q_eff = elc.delta_mid_bot * p_q;
          solver.assign_charge(q_eff, {p_pos[0], p_pos[1], -p_pos[2]}, true);
        }
        if (p_pos[2] > (elc.box_h - elc.space_layer)) {
          auto const q_eff = elc.delta_mid_top * p_q;
          solver.assign_charge(
              q_eff, {p_pos[0], p_pos[1], 2. * elc.box_h - p_pos[2]}, true);
        }
      }
    }
  }
}

template <ChargeProtocol protocol, typename combined_range>
void modify_p3m_sums(elc_data const &elc, CoulombP3M &solver,
                     combined_range const &p_q_pos_range) {

  // TODO possibly unused function
  auto local_n = std::size_t{0u};
  auto local_q2 = 0.0;
  auto local_q = 0.0;
  for (auto zipped : p_q_pos_range) {
    auto const p_q = boost::get<0>(zipped);
    auto const &p_pos = boost::get<1>(zipped);
    if (p_q != 0.) {
      auto const p_z = p_pos[2];

      if (protocol == ChargeProtocol::BOTH or
          protocol == ChargeProtocol::REAL) {
        local_n++;
        local_q2 += Utils::sqr(p_q);
        local_q += p_q;
      }

      if (protocol == ChargeProtocol::BOTH or
          protocol == ChargeProtocol::IMAGE) {
        if (p_z < elc.space_layer) {
          local_n++;
          local_q2 += Utils::sqr(elc.delta_mid_bot * p_q);
          local_q += elc.delta_mid_bot * p_q;
        }

        if (p_z > (elc.box_h - elc.space_layer)) {
          local_n++;
          local_q2 += Utils::sqr(elc.delta_mid_top * p_q);
          local_q += elc.delta_mid_top * p_q;
        }
      }
    }
  }

  auto global_n = std::size_t{0u};
  auto global_q2 = 0.;
  auto global_q = 0.;
  boost::mpi::all_reduce(comm_cart, local_n, global_n, std::plus<>());
  boost::mpi::all_reduce(comm_cart, local_q2, global_q2, std::plus<>());
  boost::mpi::all_reduce(comm_cart, local_q, global_q, std::plus<>());
  solver.count_charged_particles_elc(global_n, global_q2, Utils::sqr(global_q));
}

double ElectrostaticLayerCorrection::long_range_energy() const {
  auto const &system = get_system();

  std::cout << "[ELC] === long_range_energy() begin ===" << std::endl;

  auto const energy = std::visit(
      [this, &system](auto const &solver_ptr) {
        auto &solver = *solver_ptr;
        auto const particles = system.cell_structure->local_particles();
        auto const &box_geo = *system.box_geo;

        auto p_q_range = ParticlePropertyRange::charge_range(particles);
        auto p_pos_range = ParticlePropertyRange::pos_range(particles);
        auto p_q_pos_range = boost::combine(p_q_range, p_pos_range);

        // assign the original charges (they may not have been assigned yet)
        solver.charge_assign();

        if (!elc.dielectric_contrast_on) {
          auto const e = solver.long_range_energy();
          std::cout << std::setprecision(15)
                    << "[ELC] long_range_energy (no dielectric contrast): "
                    << "P3M long_range_energy=" << e << std::endl;
          return e;
        }

        auto energy = 0.;

        auto const e_half_p3m = 0.5 * solver.long_range_energy();
        energy += e_half_p3m;
        std::cout << std::setprecision(15)
                  << "[ELC] long_range_energy: 0.5*P3M_real=" << e_half_p3m
                  << std::endl;

        auto const e_self = 0.5 * elc.dielectric_layers_self_energy(solver, box_geo, particles);
        energy += e_self;
        std::cout << std::setprecision(15)
                  << "[ELC] long_range_energy: 0.5*dielectric_layers_self_energy="
                  << e_self << std::endl;

        // assign both original and image charges
        charge_assign<ChargeProtocol::BOTH>(elc, solver, p_q_pos_range);
        modify_p3m_sums<ChargeProtocol::BOTH>(elc, solver, p_q_pos_range);
        auto const e_both = 0.5 * solver.long_range_energy();
        energy += e_both;
        std::cout << std::setprecision(15)
                  << "[ELC] long_range_energy: +0.5*P3M_BOTH (real+image)="
                  << e_both << std::endl;

        // assign only the image charges now
        charge_assign<ChargeProtocol::IMAGE>(elc, solver, p_q_pos_range);
        modify_p3m_sums<ChargeProtocol::IMAGE>(elc, solver, p_q_pos_range);
        auto const e_image = 0.5 * solver.long_range_energy();
        energy -= e_image;
        std::cout << std::setprecision(15)
                  << "[ELC] long_range_energy: -0.5*P3M_IMAGE=" << e_image
                  << std::endl;

        // restore modified sums
        modify_p3m_sums<ChargeProtocol::REAL>(elc, solver, p_q_pos_range);

        std::cout << std::setprecision(15)
                  << "[ELC] long_range_energy: subtotal (P3M parts)=" << energy
                  << std::endl;

        return energy;
      },
      base_solver);

  auto const elc_correction = calc_energy();
  auto const total = energy + elc_correction;

  std::cout << std::setprecision(15)
            << "[ELC] long_range_energy: P3M_subtotal=" << energy
            << ", ELC_correction (calc_energy)=" << elc_correction
            << ", grand_total=" << total
            << std::endl;
  std::cout << "[ELC] === long_range_energy() end ===" << std::endl;

  return total;
}

void ElectrostaticLayerCorrection::add_long_range_forces() const {
  auto const &system = get_system();

  std::cout << "[ELC] === add_long_range_forces() begin ===" << std::endl;

  std::visit(
      [this, &system](auto const &solver_ptr) {
        auto const particles = system.cell_structure->local_particles();
        auto &solver = *solver_ptr;
        auto p_q_range = ParticlePropertyRange::charge_range(particles);
        auto p_pos_range = ParticlePropertyRange::pos_range(particles);
        auto p_q_pos_range = boost::combine(p_q_range, p_pos_range);
        if (elc.dielectric_contrast_on) {
          auto const &box_geo = *system.box_geo;
          modify_p3m_sums<ChargeProtocol::BOTH>(elc, solver, p_q_pos_range);
          charge_assign<ChargeProtocol::BOTH>(elc, solver, p_q_pos_range);
          elc.dielectric_layers_self_forces(solver, box_geo, particles);
        } else {
          solver.charge_assign();
        }
        solver.add_long_range_forces();

        // Print forces after P3M long range
        std::cout << "[ELC] Forces after solver.add_long_range_forces():"
                  << std::endl;
        for (auto const &p : particles) {
          std::cout << std::setprecision(15)
                    << "[ELC]   Particle id=" << p.id()
                    << ": Fx=" << p.force()[0]
                    << ", Fy=" << p.force()[1]
                    << ", Fz=" << p.force()[2]
                    << std::endl;
        }

        if (elc.dielectric_contrast_on) {
          modify_p3m_sums<ChargeProtocol::REAL>(elc, solver, p_q_pos_range);
        }
      },
      base_solver);

  add_force();

  // Print final forces after ELC add_force
  auto const particles = system.cell_structure->local_particles();
  std::cout << "[ELC] Final forces after add_long_range_forces():" << std::endl;
  for (auto const &p : particles) {
    std::cout << std::setprecision(15)
              << "[ELC]   Particle id=" << p.id()
              << ": Fx=" << p.force()[0]
              << ", Fy=" << p.force()[1]
              << ", Fz=" << p.force()[2]
              << std::endl;
  }
  std::cout << "[ELC] === add_long_range_forces() end ===" << std::endl;
}

#endif // ESPRESSO_P3M