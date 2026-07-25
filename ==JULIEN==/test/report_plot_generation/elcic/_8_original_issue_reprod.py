import unittest as ut

import espressomd
import espressomd.electrostatics
import numpy as np


class ELC_vs_MMM2D_neutral(ut.TestCase):
    # Handle to espresso system

    system = espressomd.System(box_l=[1.0, 1.0, 1.0])
    acc = 1e-6
    elc_gap = 8.0
    box_l = 10.0
    bl2 = box_l * 0.5
    system.time_step = 0.01
    system.cell_system.skin = 0.1

    def test_elc_vs_mmm2d(self):
        elc_param_sets = {
            "inert": {
                "gap_size": self.elc_gap,
                "maxPWerror": self.acc,
                "check_neutrality": False,
            },
            "dielectric": {
                "gap_size": self.elc_gap,
                "maxPWerror": self.acc,
                "delta_mid_bot": 0.1,
                "delta_mid_top": 0.9,
                "check_neutrality": False,
            },
            "const_pot_0": {
                "gap_size": self.elc_gap,
                "maxPWerror": self.acc,
                "const_pot": True,
                "pot_diff": 0.0,
                "check_neutrality": False,
            },
            "const_pot_1": {
                "gap_size": self.elc_gap,
                "maxPWerror": self.acc,
                "const_pot": True,
                "pot_diff": 1.0,
                "check_neutrality": False,
            },
            "const_pot_m1": {
                "gap_size": self.elc_gap,
                "maxPWerror": self.acc,
                "const_pot": True,
                "pot_diff": -1.0,
                "check_neutrality": False,
            },
        }

        case = "const_pot_0"

        # ELC
        self.system.box_l = [self.box_l, self.box_l, self.box_l + self.elc_gap]
        self.system.use_verlet_lists = True
        self.system.periodicity = [True, True, True]

        q = 3.0
        non_neutral_fac = 3.0

        self.system.part.add(id=0, pos=(5.0, 5.0, 5.0), q=-non_neutral_fac * q)
        self.system.part.add(id=1, pos=(2.0, 2.0, 5.0), q=q / 3.0)
        self.system.part.add(id=2, pos=(2.0, 5.0, 2.0), q=q / 3.0)
        self.system.part.add(id=3, pos=(5.0, 2.0, 7.0), q=q / 3.0)

        p3m = espressomd.electrostatics.P3M(
            prefactor=1.0, accuracy=self.acc, check_neutrality=False
        )

        elc = espressomd.electrostatics.ELC(actor=p3m, **elc_param_sets[case])
        self.system.electrostatics.solver = elc
        elc_res = {}

        elc_res[case] = self.scan()

        np.savetxt("data.dat", (elc_res[case]))
        print(elc_res[case])

    def scan(self):
        n = 100
        d = 0.05
        res = []
        for i in range(n + 1):
            z = self.box_l - d - 1.0 * i / n * (self.box_l - 2 * d)
            self.system.part.by_id(0).pos = [self.bl2, self.bl2, z]
            self.system.integrator.run(0)
            energy = self.system.analysis.energy()
            m = [z]
            m.extend(self.system.part.by_id(0).f)
            m.append(energy["coulomb"])
            res.append(m)

        return res


if __name__ == "__main__":
    ut.main()
