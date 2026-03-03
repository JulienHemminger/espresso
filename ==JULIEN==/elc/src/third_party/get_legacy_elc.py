# %%
import espressomd # type: ignore
import espressomd.electrostatics # type: ignore

# %%
def get_legacy_elc_energy(actor, gap_size, pw_error, system):
    elc_legacy = espressomd.electrostatics.ELC(
        actor=actor, 
        gap_size=gap_size, 
        maxPWerror=pw_error,
        check_neutrality=False,
        neutralize=False,
    )
    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    return system.analysis.energy()['total']

def get_legacy_elc_forces(actor, gap_size, pw_error, system):
    pass
    
