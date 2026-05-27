import signal
import espressomd
import espressomd.electrostatics
import time

# Define a custom exception for the timeout
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def get_legacy_energy(system, params_dict, timeout_duration_sec=30, timeout_return_value=1e3):
    system.electrostatics.clear()

    # Register the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule the alarm
    signal.alarm(timeout_duration_sec)

    try:
        gap_size      = params_dict["gap_size"]
        pw_error      = params_dict["pw_error"]
        prefactor     = params_dict.get("prefactor", 1.0)
        delta_mid_top = params_dict.get("delta_mid_top")
        delta_mid_bot = params_dict.get("delta_mid_bot")
        
        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False,
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
        if delta_mid_top == +1 and delta_mid_bot == +1:
            args["const_pot"] = True
        
        elc_legacy = espressomd.electrostatics.ELC(**args)

        system.electrostatics.solver = elc_legacy
        system.integrator.run(0)
        
        energy = system.analysis.energy()["total"]
        system.electrostatics.clear()
        
        # Disable the alarm if we finished on time
        signal.alarm(0)
        return energy

    except TimeoutException:
        print(f"--- WARNING: ELC timed out after {timeout_duration_sec}s. ---")
        try:
            system.electrostatics.clear()
        except:
            pass
        return timeout_return_value
    
    except Exception as e:
        signal.alarm(0) # Disable alarm on other errors
        print(f"--- ERROR: {e} ---")
        system.electrostatics.clear()
        return timeout_return_value