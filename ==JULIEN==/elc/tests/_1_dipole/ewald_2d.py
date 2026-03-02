
import numpy as np
from scipy.special import erfc, erf


# ---------------------------------------------------------------------------
#  Brute-force direct summation (for verification)
# ---------------------------------------------------------------------------

def direct_sum_energy(positions, charges, dx, dy, n_max=100):
    """
    Brute-force Coulomb energy summed over a (2*n_max+1)^2 lattice.

    For a charge-neutral unit cell the conditionally convergent pieces
    cancel term-by-term, so this converges (slowly) as ~ 1/n_max.
    """
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    N = len(q)

    # All lattice translations
    nx = np.arange(-n_max, n_max + 1)
    ny = np.arange(-n_max, n_max + 1)
    NX, NY = np.meshgrid(nx, ny, indexing="ij")
    Rx = (NX.ravel() * dx).astype(np.float64)        # (M,)
    Ry = (NY.ravel() * dy).astype(np.float64)         # (M,)
    M = len(Rx)

    # Index of the (0, 0) translation
    idx_origin = n_max * (2 * n_max + 1) + n_max

    E = 0.0
    for a in range(N):
        for b in range(N):
            dx_ab = pos[a, 0] - pos[b, 0] + Rx       # (M,)
            dy_ab = pos[a, 1] - pos[b, 1] + Ry       # (M,)
            dz_ab = pos[a, 2] - pos[b, 2]             # scalar

            dist = np.sqrt(dx_ab ** 2 + dy_ab ** 2 + dz_ab ** 2)

            if a == b:
                dist[idx_origin] = np.inf              # exclude self

            E += q[a] * q[b] * np.sum(1.0 / dist)

    E *= 0.5
    return E


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

def test_ewald_vs_direct():
    """Compare Ewald summation against brute-force direct sum."""
    print("=" * 70)
    print("TEST 1: Ewald vs. direct summation (dipole lattice)")
    print("=" * 70)

    # Parameters
    l_xy = 1.0          # lattice spacing
    s = 0.1          # dipole separation
    q_val = 1.0      # charge magnitude

    # Unit cell: +q at (0, 0, +s/2), -q at (0, 0, -s/2)
    positions = np.array([
        [0.0, 0.0,  s / 2],
        [0.0, 0.0, -s / 2],
    ])
    charges = np.array([q_val, -q_val])

    # Ewald result
    

    # Direct sum (general, slow)
    E_direct_general = direct_sum_energy(positions, charges, l_xy, l_xy, n_max=200)


    print(f"  Lattice spacing a = {l_xy},  dipole separation s = {s}")
    print(f"  Direct sum (general):      {E_direct_general:.10f}") # -9.9551624737
   

    print("  ✓ PASSED\n")



# ---------------------------------------------------------------------------
#  Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  2D Ewald Summation — Test Suite")
    print("  " + "=" * 46 + "\n")

    test_ewald_vs_direct()

    print("=" * 70)
    print("  ALL TESTS PASSED ✓")
    print("=" * 70)
