# testsuite/samples/test_visualization_elc.py
"""
./pypresso /home/main/Documents/Career/1_Studium/ESPRESSO/Cursor/espresso/==JULIEN==/_problemsolving/espresso/tutorials/test_visualization_elc.py
"""

import unittest as ut

import importlib_wrapper


def disable_GUI(code):
    # integrate without visualizer
    breakpoint = "visualizer.run(1)"
    assert breakpoint in code
    code = code.replace(breakpoint, "steps=1\nsystem.integrator.run(steps)", 1)
    return code


sample, skipIfMissingFeatures = importlib_wrapper.configure_and_import(
    "/home/main/Documents/Career/1_Studium/ESPRESSO/Cursor/espresso/samples/visualization_elc.py",
    substitutions=disable_GUI,
    steps=5000,
)


@skipIfMissingFeatures
class Sample(ut.TestCase):
    system = sample.system

    def test_dipole_moment(self):
        import espressomd.observables

        obs = espressomd.observables.DipoleMoment(ids=self.system.part.all().id)
        dipm = obs.calculate()
        self.assertLess(dipm[2], 0, msg="charges moved in the wrong direction")
        # the dipole moment should be the strongest along the z-axis
        self.assertGreater(abs(dipm[2]), abs(dipm[0]))
        self.assertGreater(abs(dipm[2]), abs(dipm[1]))


if __name__ == "__main__":
    ut.main()
# OK (passes)
