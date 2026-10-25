// Ketju-integrator (MSTAR)
// Copyright (C) 2020-2022 Matias Mannerkoski and contributors.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
// Modified by Ulrich Steinwandel, 2026: rewritten in C++ with Kokkos support (GPU).
// Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// C++ port of src/bh_mergers.c (BH mergers, the contact collisions of non-PN particles and their energy booking) and
// of the PN c.o.m. velocity in src/pn.c.

#include "mergers.hpp"

#include <cmath>
#include <utility>

#include "kernels.hpp"
#include "mstar/mstar.hpp"

namespace mstar {
namespace detail {

namespace {
constexpr double PI = 3.14159265358979323846;
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - b[1] * a[2], a[2] * b[0] - b[2] * a[0], a[0] * b[1] - b[0] * a[1]};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 unit(const Vec3& a) {
    const double n = norm(a);
    return {a[0] / n, a[1] / n, a[2] / n};
}
}  // namespace

// Zlochower & Lousto 2015, PhRvD 92, 024022 (arXiv:1503.07536, corrected version); see the notes in bh_mergers.c
RemnantProperties bh_merger_remnant(const MergerInput& in) {
    RemnantProperties out{};
    const double G = in.G, c = in.c, m1 = in.m1, m2 = in.m2, m = m1 + m2;
    Vec3 rv, vv;
    for (int i = 0; i < 3; i++) { rv[i] = in.x1[i] - in.x2[i]; vv[i] = in.v1[i] - in.v2[i]; }
    Vec3 L = cross(rv, vv);
    for (int i = 0; i < 3; ++i) L[i] *= m1 * m2 / m;
    if (L[0] == 0 && L[1] == 0 && L[2] == 0) {   // head-on: no GW emission to speak of
        out.m = m;
        for (int i = 0; i < 3; ++i) { out.vkick[i] = 0; out.S[i] = in.S1[i] + in.S2[i]; }
        return out;
    }
    const Vec3 Lhat = unit(L);
    const Vec3 n0 = unit(rv);
    const Vec3 m0 = cross(Lhat, n0);
    Vec3 m59;
    {
        const double angle = -59 * PI / 180.0;
        for (int i = 0; i < 3; i++) m59[i] = n0[i] * std::cos(angle) + m0[i] * std::sin(angle);
    }
    Vec3 J;
    for (int i = 0; i < 3; i++) J[i] = in.S1[i] + in.S2[i] + L[i];
    const Vec3 Jhat = unit(J);
    Vec3 S1, S2;
    for (int i = 0; i < 3; i++) { S1[i] = in.S1[i] * c / G; S2[i] = in.S2[i] * c / G; }
    const double delta_m = (m1 - m2) / m;
    const double eta = (1 - delta_m * delta_m) / 4;
    Vec3 S_tilde, Delta_tilde, S0_tilde;
    for (int i = 0; i < 3; i++) {
        S_tilde[i] = (S1[i] + S2[i]) / (m * m);
        Delta_tilde[i] = (S2[i] / m2 - S1[i] / m1) / m;
        S0_tilde[i] = S_tilde[i] + 0.5 * delta_m * Delta_tilde[i];
    }
    const double Delta_tilde_para = dot(Delta_tilde, Lhat);
    const double Delta_tilde_ortho = std::hypot(dot(Delta_tilde, n0), dot(Delta_tilde, m0));
    const double S_tilde_para = dot(S_tilde, Lhat);
    const double S_tilde_ortho = std::hypot(dot(S_tilde, n0), dot(S_tilde, m0));
    const double S0_tilde_para = dot(S0_tilde, Lhat);
    auto P = [](double x, int k) { return std::pow(x, k); };

    // Eq. (37), with the (4 eta)^2 leading factor (see bh_mergers.c)
    const double v_para =
        P(4 * eta, 2)
        * (dot(Delta_tilde, n0)
               * (3678. - 2475. * P(delta_m, 2) + 4962. * S0_tilde_para + 7170. * P(S0_tilde_para, 2)
                  + 12050. * P(S0_tilde_para, 3))
           + dot(S0_tilde, m59)
                 * (Delta_tilde_para * (4315. - 1262. * P(delta_m, 2) + 15970. * S0_tilde_para) - 2256. * delta_m
                    - 2231. * delta_m * S0_tilde_para));
    // Eq. (39)
    double v_ortho2 =
        P(4 * eta, 4)
            * (2.106e5 * P(Delta_tilde_para, 2) + 4.967e5 * Delta_tilde_para * delta_m
               - 2.116e5 * P(Delta_tilde_para, 3) * delta_m - 5.037e5 * P(Delta_tilde_para, 2) * P(delta_m, 2)
               - 1.269e5 * Delta_tilde_para * P(delta_m, 3) - 3.384e5 * P(Delta_tilde_para, 2) * S0_tilde_para
               - 6.440e5 * P(delta_m, 2) * S0_tilde_para + 2.138e6 * P(Delta_tilde_para, 2) * P(S0_tilde_para, 2)
               - 4.905e6 * Delta_tilde_para * delta_m * P(S0_tilde_para, 2)
               - 1.100e6 * P(delta_m, 2) * P(S0_tilde_para, 2)
               - 1.024e7 * P(Delta_tilde_para, 2) * P(S0_tilde_para, 3))
        + P(1.2e4 * eta * eta * delta_m * (1 - 0.93 * eta), 2);
    if (v_ortho2 < 0) v_ortho2 = 0;
    const double v_ortho = std::sqrt(v_ortho2);
    const double km_per_s = c / 2.99792458e5;
    for (int i = 0; i < 3; i++) out.vkick[i] = km_per_s * (v_para * Lhat[i] + v_ortho * n0[i]);

    // Kerr ISCO quantities for chi = S_tilde_para
    const double chi = S_tilde_para;
    const double sign_chi = chi == 0 ? 0 : (chi > 0 ? 1 : -1);
    const double Z1 = 1 + std::pow(1 - chi * chi, 1. / 3) * (std::pow(1 + chi, 1. / 3) + std::pow(1 - chi, 1. / 3));
    const double Z2 = std::sqrt(3 * chi * chi + Z1 * Z1);
    const double r_isco = 3 + Z2 - sign_chi * std::sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));
    const double L_isco = 2. / std::sqrt(3 * r_isco) * (3 * std::sqrt(r_isco) - 2 * chi);
    const double E_isco = std::sqrt(1 - 2. / (3 * r_isco));
    {   // mass loss, Eqs. (30)-(34), E_c term only
        const double k2a = -0.024, k2d = 0, k3a = -0.055, k3b = 0.0, k3d = -0.019, k4a = -0.119, k4b = 0.005, k4e = 0,
                     k4f = 0.035, k4g = 0.022;
        const double E_A = 0, E_B = 0.59, E_D = 0, E_E = -0.51, E_F = 0.056, E_G = -0.073, E_H = 0;
        const double e1 = 0.0356, e2 = 0.096, e3 = 0.122, eps1 = 0.0043, eps2 = 0.0050, eps3 = -0.009;
        const double E_HU = 0.0025829 - 0.0773079 / (2 * S0_tilde_para - 1.693959);
        const double Ec_para =
            P(4 * eta, 2)
                * (E_HU + k2a * delta_m * Delta_tilde_para + 0.000743 * P(Delta_tilde_para, 2) + k2d * P(delta_m, 2)
                   + k3a * delta_m * Delta_tilde_para * S_tilde_para + k3b * S_tilde_para * P(Delta_tilde_para, 2)
                   + k3d * P(delta_m, 2) * S_tilde_para + k4a * delta_m * Delta_tilde_para * P(S_tilde_para, 2)
                   + k4b * delta_m * P(Delta_tilde_para, 3) + 0.000124 * P(Delta_tilde_para, 4)
                   + k4e * P(Delta_tilde_para, 2) * P(S_tilde_para, 2) + k4f * P(delta_m, 4)
                   + k4g * P(delta_m, 3) * Delta_tilde_para)
            + P(delta_m, 6) * eta * (1 - E_isco);
        const double Ec =
            Ec_para
            + P(4 * eta, 2)
                  * (P(S_tilde_ortho, 2) * (e1 + e2 * S_tilde_para + e3 * P(S_tilde_para, 2))
                     + P(Delta_tilde_ortho, 2) * (eps1 + eps2 * S_tilde_para + eps3 * P(S_tilde_para, 2))
                     + P(delta_m, 2) * P(S_tilde_ortho, 2) * (E_A + S_tilde_para * E_B)
                     + P(delta_m, 2) * P(Delta_tilde_ortho, 2) * (E_D + S_tilde_para * E_E)
                     + E_F * delta_m * Delta_tilde_ortho * S_tilde_ortho
                     + E_G * P(Delta_tilde_para, 2) * P(Delta_tilde_ortho, 2)
                     + E_H * P(Delta_tilde_para, 2) * P(S_tilde_ortho, 2));
        out.m = m * (1 - Ec);
    }
    {   // remnant spin, Eqs. (35)-(36), A_c term only, along J
        const double L0 = 0.686710, L1 = 0.613247, L2a = -0.145427, L2b = -0.115689, L2c = -0.005254,
                     L2d = 0.801838, L3a = -0.073839, L3b = 0.004759, L3c = -0.078377, L3d = 1.585809,
                     L4a = -0.003050, L4b = -0.002968, L4c = 0.004364, L4d = -0.047204, L4e = -0.053099,
                     L4f = 0.953458, L4g = -0.067998, L4h = 0.001629, L4i = -0.066693;
        const double A_A = 3.0, A_B = -7.3, A_D = -2.0, A_E = 5.1, A_F = 0, A_G = -2.8, A_H = 5.1;
        const double a1 = 0.8401, a2 = -0.328, a3 = -0.61, zeta1 = -0.0209, zeta2 = -0.038, zeta3 = 0.04;
        const double alpha_align =
            P(4 * eta, 2)
                * (L0 + L1 * S_tilde_para + L2a * delta_m * Delta_tilde_para + L2b * P(S_tilde_para, 2)
                   + L2c * P(Delta_tilde_para, 2) + L2d * P(delta_m, 2)
                   + L3a * Delta_tilde_para * S_tilde_para * delta_m + L3b * S_tilde_para * P(Delta_tilde_para, 2)
                   + L3c * P(S_tilde_para, 3) + L3d * S_tilde_para * P(delta_m, 2)
                   + L4a * Delta_tilde_para * P(S_tilde_para, 2) * delta_m + L4b * P(Delta_tilde_para, 3) * delta_m
                   + L4c * P(Delta_tilde_para, 4) + L4d * P(S_tilde_para, 4)
                   + L4e * P(Delta_tilde_para, 2) * P(S_tilde_para, 2) + L4f * P(delta_m, 4)
                   + L4g * Delta_tilde_para * P(delta_m, 3) + L4h * P(Delta_tilde_para * delta_m, 2)
                   + L4i * P(S_tilde_para * delta_m, 2))
            + S_tilde_para * (1 + 8 * eta) * P(delta_m, 4) + eta * L_isco * P(delta_m, 6);
        const double Ac =
            P(alpha_align, 2)
            + P(4 * eta, 2)
                  * (P(S_tilde_ortho, 2) * (a1 + a2 * S_tilde_para + a3 * P(S_tilde_para, 2))
                     + P(Delta_tilde_ortho, 2) * (zeta1 + zeta2 * S_tilde_para + zeta3 * P(S_tilde_para, 2))
                     + P(delta_m, 2) * P(S_tilde_ortho, 2) * (A_A + A_B * S_tilde_para)
                     + P(delta_m, 2) * P(Delta_tilde_ortho, 2) * (A_D + A_E * S_tilde_para)
                     + A_F * delta_m * Delta_tilde_ortho * S_tilde_ortho
                     + A_G * P(Delta_tilde_para, 2) * P(Delta_tilde_ortho, 2)
                     + A_H * P(Delta_tilde_para, 2) * P(S_tilde_ortho, 2))
            + P(delta_m, 6) * (1 + 12 * eta) * P(S_tilde_ortho, 2);
        double spin_mag = std::sqrt(Ac);
        if (spin_mag > 1.) spin_mag = 1.;
        for (int i = 0; i < 3; i++) out.S[i] = (G / c) * out.m * out.m * spin_mag * Jhat[i];
    }
    return out;
}

namespace {
// binary_PN_CoM_momentum_single_contribution (pn.c): momentum contribution of particle i; returns its mass corrected
// with the Newtonian binding energy
double pn_com_momentum_single(const std::vector<double>& mass, const std::vector<double>& pos,
                              const std::vector<double>& vel, const std::vector<double>& spin, int i, int j, double G,
                              double c, unsigned flags, Vec3& p_out) {
    const double G2 = G * G, G3 = G2 * G, c2 = c * c, c4 = c2 * c2, c6 = c4 * c2;
    const Vec3 vi = {vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]};
    const Vec3 vj = {vel[3 * j], vel[3 * j + 1], vel[3 * j + 2]};
    const double mi = mass[i], mi2 = mi * mi, mi3 = mi2 * mi;
    const double vi2 = dot(vi, vi), vi4 = vi2 * vi2;
    Vec3 rv, vv;
    for (int k = 0; k < 3; k++) { rv[k] = pos[3 * i + k] - pos[3 * j + k]; vv[k] = vi[k] - vj[k]; }
    const double rij = norm(rv);
    const Vec3 nv = {rv[0] / rij, rv[1] / rij, rv[2] / rij};
    const double rij2 = rij * rij, rij3 = rij2 * rij;
    const double mj = mass[j], mj2 = mj * mj, mj3 = mj2 * mj;
    const double vivj = dot(vi, vj), vj2 = dot(vj, vj), vj4 = vj2 * vj2;
    const double vjnij = dot(vj, nv), vjnij2 = vjnij * vjnij, vjnij3 = vjnij2 * vjnij, vjnij4 = vjnij2 * vjnij2;
    const double vinij = dot(vi, nv), vinij2 = vinij * vinij, vinij3 = vinij2 * vinij, vinij4 = vinij2 * vinij2;
    double vi_fac = mi, nij_fac = 0;
    if (flags & PN_1_0) {
        nij_fac += -G * mi * mj / (2. * rij) * vinij / c2;
        vi_fac += (.5 * mi * vi2 - G * mi * mj / (2. * rij)) / c2;
    }
    if (flags & PN_2_0) {
        nij_fac += (G2 * mi2 * mj / rij2 * (29. / 4. * vinij - 9. / 4. * vjnij)
                    + G * mi * mj / rij
                          * (3. / 8. * vinij3 + 3. / 8. * vinij2 * vjnij - 9. / 8. * vinij * vi2
                             - 7. / 8. * vjnij * vi2 + 7. / 4. * vinij * vivj))
                   / c4;
        vi_fac += (-3. * G2 * mi2 * mj / rij2 + 7. * G2 * mi * mj2 / (2. * rij2) + 3. / 8. * mi * vi4
                   + G * mi * mj / rij
                         * (13. / 8. * vinij2 - .25 * vinij * vjnij - 13. / 8. * vjnij2 + 5. / 8. * vi2
                            - 7. / 4. * vivj + 7. / 8. * vj2))
                  / c4;
    }
    if (flags & PN_2_5) {
        const double vij2 = dot(vv, vv);
        nij_fac += 4. * G2 * mi2 * mj / (5. * c4 * c * rij2) * (vij2 - 2. * G * mi / rij);
    }
    if (flags & PN_3_0) {
        nij_fac +=
            (G2 * mi2 * mj / rij2
                 * (-45. / 8. * vinij3 + 59. / 8. * vinij2 * vjnij - 179. / 8. * vinij * vjnij2 + 247. / 24. * vjnij3
                    + 135. / 8. * vinij * vi2 - 87. / 8. * vjnij * vi2 - 53. / 2. * vinij * vivj
                    + 87. / 4. * vjnij * vivj + 53. / 4. * vinij * vj2 - 12. * vjnij * vj2)
             + G * mi * mj / rij
                   * (-5. / 16. * vinij3 * vinij2 - 5. / 16. * vinij4 * vjnij - 5. / 16. * vinij3 * vjnij2
                      + 19. / 16. * vinij3 * vi2 + 15. / 16. * vinij2 * vjnij * vi2 + 3. / 4. * vinij * vjnij2 * vi2
                      + 5. / 8. * vjnij3 * vi2 - 21. / 16. * vinij * vi4 - 11. / 16. * vjnij * vi4
                      - 5. / 4. * vinij3 * vivj - 9. / 8. * vinij2 * vjnij * vivj + 15. / 8. * vinij * vi2 * vivj
                      + vjnij * vi2 * vivj - 1. / 8. * vinij * vivj * vivj - 15. / 16. * vinij * vi2 * vj2)
             - 175. * G3 * mi2 * mj2 / (8. * rij3) * vinij
             + G3 * mi3 * mj / rij3 * (-46517. / 840. * vinij + 34547. / 840. * vjnij + 22. / 3. * (vinij - vjnij)))
            / c6;
        vi_fac += (5. * mi * vi2 * vi4 / 16.
                   + G2 * mi2 * mj / rij2
                         * (-39. / 4. * vinij2 + 22. * vinij * vjnij - 21. / 4. * vjnij2 - 2. * vi2 + vivj - .5 * vj2)
                   + G2 * mi * mj2 / rij2
                         * (23. / 4. * vinij2 - 101. / 4. * vinij * vjnij + 17. * vjnij2 + 9. / 4. * vi2 - vivj
                            + .5 * vj2)
                   + G * mi * mj / rij
                         * (-19. / 16. * vinij4 + 1. / 4. * vinij3 * vjnij + 3. / 16. * vinij2 * vjnij2
                            + 1. / 8. * vinij * vjnij3 + 19. / 16. * vjnij4 + 45. / 16. * vinij2 * vi2
                            - 5. / 8. * vinij * vjnij * vi2 - 2. * vjnij2 * vi2 + 23. / 16. * vi4
                            - 3. / 4. * vinij2 * vivj + 3. / 4. * vinij * vjnij * vivj + 19. / 8. * vjnij2 * vivj
                            - 31. / 8. * vi2 * vivj + 17. / 8. * vivj * vivj + 3. / 8. * vinij2 * vj2
                            - .5 * vinij * vjnij * vj2 - 45. / 16. * vjnij2 * vj2 + 31. / 16. * vi2 * vj2
                            - 3. * vivj * vj2 + 19. / 16. * vj4)
                   - 19. * G3 * mi2 * mj2 / (8. * rij3) + G3 * mi3 * mj / rij3 * 20407. / 2520.
                   - G3 * mi * mj3 / rij3 * 21667. / 2520.)
                  / c6;
    }
    Vec3 spin_term = {0, 0, 0};
    if (flags & PN_1_5_SPIN) {
        Vec3 acc, Si = {spin[3 * i], spin[3 * i + 1], spin[3 * i + 2]};
        for (int k = 0; k < 3; ++k) acc[k] = -G * mj / rij2 * nv[k];
        spin_term = cross(acc, Si);
    }
    for (int k = 0; k < 3; ++k) p_out[k] = vi_fac * vi[k] + nij_fac * nv[k] + spin_term[k] / c2;
    return mi + (.5 * mi * vi2 - G * mi * mj / (2 * rij)) / c2;
}
}  // namespace

Vec3 binary_pn_com_velocity(const std::vector<double>& mass, const std::vector<double>& pos,
                            const std::vector<double>& vel, const std::vector<double>& spin, int i, int j, double G,
                            double c, unsigned flags) {
    Vec3 p1, p2, v;
    const double m1 = pn_com_momentum_single(mass, pos, vel, spin, i, j, G, c, flags, p1);
    const double m2 = pn_com_momentum_single(mass, pos, vel, spin, j, i, G, c, flags, p2);
    for (int k = 0; k < 3; ++k) v[k] = (p1[k] + p2[k]) / (m1 + m2);
    return v;
}


namespace {
double pair_distance(const System& o, int i, int j) {
    double r2 = 0;
    for (int k = 0; k < 3; ++k) { const double dr = o.pos[3 * i + k] - o.pos[3 * j + k]; r2 += dr * dr; }
    return std::sqrt(r2);
}
// particle j leaves the arrays (data to its right shift left); its extra_index moves behind the remaining particles
void remove_particle(System& o, int j) {
    const int n = o.num_particles;
    const int gone = o.extra_index[j];
    o.mass.erase(o.mass.begin() + j);
    o.pos.erase(o.pos.begin() + 3 * j, o.pos.begin() + 3 * j + 3);
    o.vel.erase(o.vel.begin() + 3 * j, o.vel.begin() + 3 * j + 3);
    if (j < o.num_pn_particles) o.spin.erase(o.spin.begin() + 3 * j, o.spin.begin() + 3 * j + 3);
    for (int q = j; q + 1 < n; q++) o.extra_index[q] = o.extra_index[q + 1];
    o.extra_index[n - 1] = gone;
    if ((int)o.collision_radius.size() == n) o.collision_radius.erase(o.collision_radius.begin() + j);
    auto& sc = o.stop_collisions;
    if ((int)sc.radius.size() == n) sc.radius.erase(sc.radius.begin() + j);
    if ((int)sc.is_bh.size() == n) sc.is_bh.erase(sc.is_bh.begin() + j);
    o.num_particles -= 1;
    if (j < o.num_pn_particles) o.num_pn_particles -= 1;
}
}  // namespace

bool bh_merger_due(const System& o) {
    if (o.num_pn_particles < 2) return false;
    const double fac = o.options.bh_merger_distance_in_Rs * 2 * o.constants.G / (o.constants.c * o.constants.c);
    for (int i = 0; i < o.num_pn_particles; ++i)
        for (int j = i + 1; j < o.num_pn_particles; ++j)
            if (pair_distance(o, i, j) < fac * (o.mass[i] + o.mass[j])) return true;
    return false;
}

namespace {
// merge_bhs (bh_mergers.c): PN particle j into i (i < j after ordering)
void merge_bhs(System& o, int i, int j) {
    if (j < i) std::swap(i, j);
    const double G = o.constants.G, c = o.constants.c;
    MergerInput in{G, c, o.mass[i], o.mass[j], {}, {}, {}, {}, {}, {}};
    for (int k = 0; k < 3; k++) {
        in.x1[k] = o.pos[3 * i + k]; in.x2[k] = o.pos[3 * j + k];
        in.v1[k] = o.vel[3 * i + k]; in.v2[k] = o.vel[3 * j + k];
        in.S1[k] = o.spin[3 * i + k]; in.S2[k] = o.spin[3 * j + k];
    }
    Vec3 xcom;
    for (int k = 0; k < 3; k++) xcom[k] = (in.m1 * in.x1[k] + in.m2 * in.x2[k]) / (in.m1 + in.m2);
    const Vec3 vcom = binary_pn_com_velocity(o.mass, o.pos, o.vel, o.spin, i, j, G, c, o.options.PN_flags);
    const RemnantProperties res = bh_merger_remnant(in);
    for (int k = 0; k < 3; ++k) {
        o.pos[3 * i + k] = xcom[k];
        o.vel[3 * i + k] = vcom[k] + (o.options.enable_bh_merger_kicks ? res.vkick[k] : 0);
        o.spin[3 * i + k] = res.S[k];
    }
    o.mass[i] = res.m;
    const double gc = in.G / in.c;
    MergerInfo info;
    info.t_merger = o.time;
    info.m1 = in.m1;
    info.m2 = in.m2;
    info.m_remnant = res.m;
    info.chi1 = norm(in.S1) / (gc * in.m1 * in.m1);
    info.chi2 = norm(in.S2) / (gc * in.m2 * in.m2);
    info.chi_remnant = norm(res.S) / (gc * res.m * res.m);
    info.v_kick = norm(res.vkick);
    info.extra_index1 = o.extra_index[i];
    info.extra_index2 = o.extra_index[j];
    info.is_collision = 0;
    o.mergers.push_back(info);
    remove_particle(o, j);
}

// merge_stars (bh_mergers.c): non-PN particle j into i (i < j) at their CoM, mass and momentum conserved; the heavier
// keeps slot i (its extra_index and its radius, the product gets the larger radius); the energy change is booked
void merge_stars(System& o, int i, int j) {
    std::vector<double>& m = o.mass;
    std::vector<double>& x = o.pos;
    std::vector<double>& v = o.vel;
    std::vector<double>& R = o.collision_radius;
    if (m[j] > m[i]) {   // the heavier star survives: it takes slot i (CoM state is symmetric)
        std::swap(o.extra_index[i], o.extra_index[j]);
        std::swap(R[i], R[j]);
        std::swap(m[i], m[j]);
        for (int k = 0; k < 3; ++k) {
            std::swap(x[3 * i + k], x[3 * j + k]);
            std::swap(v[3 * i + k], v[3 * j + k]);
        }
    }
    const double m1 = m[i], m2 = m[j], M = m1 + m2;
    double r2 = 0, v2 = 0;
    for (int k = 0; k < 3; ++k) {
        r2 += (x[3 * j + k] - x[3 * i + k]) * (x[3 * j + k] - x[3 * i + k]);
        v2 += (v[3 * j + k] - v[3 * i + k]) * (v[3 * j + k] - v[3 * i + k]);
    }
    // the energy MSTAR conserves: star-star pairs softened as integrated, PN pairs not
    const double G = o.constants.G, h = o.options.star_star_softening;
    const bool no_ss = o.options.no_star_star_gravity;
    const int npn = o.num_pn_particles, n = o.num_particles;
    const double E_pair = 0.5 * m1 * m2 / M * v2 - (no_ss ? 0 : G * m1 * m2 * softened_inverse_r(std::sqrt(r2), h));
    // potential energy change with the other particles as the pair moves to its CoM
    double dE_others = 0;
    {
        double xc[3];
        for (int k = 0; k < 3; ++k) xc[k] = (m1 * x[3 * i + k] + m2 * x[3 * j + k]) / M;
        for (int q = 0; q < n; ++q) {
            if (q == i || q == j) continue;
            if (no_ss && q >= npn) continue;
            const double hq = q < npn ? 0 : h;
            double d1 = 0, d2 = 0, dc = 0;
            for (int k = 0; k < 3; ++k) {
                d1 += (x[3 * q + k] - x[3 * i + k]) * (x[3 * q + k] - x[3 * i + k]);
                d2 += (x[3 * q + k] - x[3 * j + k]) * (x[3 * q + k] - x[3 * j + k]);
                dc += (x[3 * q + k] - xc[k]) * (x[3 * q + k] - xc[k]);
            }
            dE_others += G * m[q]
                         * (m1 * softened_inverse_r(std::sqrt(d1), hq) + m2 * softened_inverse_r(std::sqrt(d2), hq)
                            - M * softened_inverse_r(std::sqrt(dc), hq));
        }
    }
    // external tide: -1/2 sum m x.T.x changes by +1/2 mu r.T.r
    double dE_tide = 0;
    if (o.options.enable_external_tide) {
        const double* t = o.options.external_tidal_tensor;
        double r[3];
        for (int k = 0; k < 3; ++k) r[k] = x[3 * j + k] - x[3 * i + k];
        const double Tr[3] = {t[0] * r[0] + t[3] * r[1] + t[5] * r[2], t[3] * r[0] + t[1] * r[1] + t[4] * r[2],
                              t[5] * r[0] + t[4] * r[1] + t[2] * r[2]};
        dE_tide = 0.5 * m1 * m2 / M * (r[0] * Tr[0] + r[1] * Tr[1] + r[2] * Tr[2]);
    }
    for (int k = 0; k < 3; ++k) {
        x[3 * i + k] = (m1 * x[3 * i + k] + m2 * x[3 * j + k]) / M;
        v[3 * i + k] = (m1 * v[3 * i + k] + m2 * v[3 * j + k]) / M;
    }
    m[i] = M;
    if (R[j] > R[i]) R[i] = R[j];   // contact radius of the product: the larger one (the caller may reset it)
    MergerInfo info;
    info.t_merger = o.time;
    info.m1 = m1;
    info.m2 = m2;
    info.m_remnant = M;
    info.extra_index1 = o.extra_index[i];
    info.extra_index2 = o.extra_index[j];
    info.is_collision = 1;
    info.dE_collision = -E_pair + dE_others + dE_tide;
    o.mergers.push_back(info);
    remove_particle(o, j);
}
}  // namespace

int do_bh_mergers(System& o) {
    int num_merged = 0;
    const double fac = o.options.bh_merger_distance_in_Rs * 2 * o.constants.G / (o.constants.c * o.constants.c);
    for (int i = 0; i < o.num_pn_particles; ++i)
        for (int j = i + 1; j < o.num_pn_particles; ++j)
            if (pair_distance(o, i, j) < fac * (o.mass[i] + o.mass[j])) {
                merge_bhs(o, i, j);
                num_merged += 1;
            }
    return num_merged;
}

int star_collisions(System& o, const std::function<double(int, int)>* rv_prev, double dt_check, double* dE) {
    if ((int)o.collision_radius.size() != o.num_particles) return 0;
    int num_merged = 0;
    const double G = o.constants.G;
    const int n_entry = o.num_particles;
    bool again = true;
    while (again) {
        again = false;
        const int n = o.num_particles;
        const std::vector<double>& R = o.collision_radius;
        for (int i = o.num_pn_particles; i < n && !again; ++i) {
            if (!(R[i] > 0)) continue;
            for (int j = i + 1; j < n; ++j) {
                if (!(R[j] > 0)) continue;
                double dr[3], dv[3];
                for (int k = 0; k < 3; ++k) {
                    dr[k] = o.pos[3 * j + k] - o.pos[3 * i + k];
                    dv[k] = o.vel[3 * j + k] - o.vel[3 * i + k];
                }
                const bool use_prev = rv_prev && n == n_entry;
                if (star_contact(dr, dv, o.mass[i], o.mass[j], G, R[i] + R[j], use_prev,
                                 use_prev ? (*rv_prev)(i, j) : 0.0, dt_check)) {
                    merge_stars(o, i, j);
                    if (dE) *dE += o.mergers.back().dE_collision;
                    num_merged += 1;
                    again = true;   // indices shifted: rescan
                    break;
                }
            }
        }
    }
    return num_merged;
}

}  // namespace detail

void pair_rv(const System& S, double* rv) {
    const int n = S.num_particles;
    rv[n * n] = S.time;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += (S.pos[3 * j + k] - S.pos[3 * i + k]) * (S.vel[3 * j + k] - S.vel[3 * i + k]);
            rv[i * n + j] = s;
        }
}

int do_star_collisions(System& S, double* rv_prev) {
    if ((int)S.collision_radius.size() != S.num_particles) return 0;
    const int n = S.num_particles;
    const double dt_check = rv_prev ? S.time - rv_prev[n * n] : 0;
    std::function<double(int, int)> f = [rv_prev, n](int i, int j) { return rv_prev[i * n + j]; };
    const int num = detail::star_collisions(S, rv_prev ? &f : nullptr, dt_check, nullptr);
    if (rv_prev) pair_rv(S, rv_prev);
    return num;
}

}  // namespace mstar
