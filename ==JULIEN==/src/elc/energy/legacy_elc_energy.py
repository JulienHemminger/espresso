import espressomd
import espressomd.electrostatics
import signal

def get_legacy_elc_energy(
    system, gap_size, pw_error, prefactor=1.0, delta_mid_top=None, delta_mid_bot=None,
    duration_limit_sec=20, fallback_return_value=0
):
    # Internal function to handle the timeout signal
    def handler(signum, frame):
        raise TimeoutError("Computation exceeded the 20-second time limit.")

    # Register the signal handler and set the alarm for 20 seconds
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(duration_limit_sec)

    try:
        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
        )

        args = {
            "actor": p3m,
            "gap_size": gap_size,
            "maxPWerror": pw_error,
            "check_neutrality": False,
            "neutralize": False,
        }

        if delta_mid_top is not None:
            args["delta_mid_top"] = delta_mid_top
        if delta_mid_bot is not None:
            args["delta_mid_bot"] = delta_mid_bot
        if delta_mid_top == -1 and delta_mid_bot == -1:
            args["const_pot"] = True

        elc_legacy = espressomd.electrostatics.ELC(**args)

        system.electrostatics.solver = elc_legacy
        system.integrator.run(0)
        #print(p3m.get_params()) 
        
        # Disable the alarm if computation finishes on time
        signal.alarm(0)
        return system.analysis.energy()["total"]

    except TimeoutError as e:
        print(f"Warning: {e}")
        return fallback_return_value
    except Exception as e:
        # Catch other potential errors, but disable the alarm first
        signal.alarm(0)
        print(f"An unexpected error occurred: {e}")
        return fallback_return_value