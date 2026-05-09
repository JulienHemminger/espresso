import espressomd
import espressomd.electrostatics
from multiprocessing import Process, Queue

def _worker_get_energy(queue, system_state, gap_size, pw_error, prefactor, delta_mid_top, delta_mid_bot):
    """
    Internal worker: ESPResSo objects often cannot be pickled easily, 
    so you may need to pass necessary parameters to reconstruct the state 
    or ensure the system object is accessible.
    """
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

        if delta_mid_top is not None: args["delta_mid_top"] = delta_mid_top
        if delta_mid_bot is not None: args["delta_mid_bot"] = delta_mid_bot
        if delta_mid_top == -1 and delta_mid_bot == -1: args["const_pot"] = True

        elc_legacy = espressomd.electrostatics.ELC(**args)
        
        # Assuming system is accessible or passed
        system_state.electrostatics.solver = elc_legacy
        system_state.integrator.run(0)
        
        energy = system_state.analysis.energy()["total"]
        queue.put(energy)
    except Exception as e:
        queue.put(e)

def get_legacy_elc_energy(
    system, gap_size, pw_error, prefactor=1.0, delta_mid_top=None, delta_mid_bot=None,
    duration_limit_sec=10, fallback_return_value=0
):
    q = Queue()
    # Create a separate process
    p = Process(target=_worker_get_energy, args=(
        q, system, gap_size, pw_error, prefactor, delta_mid_top, delta_mid_bot
    ))
    
    p.start()
    # Wait for the result for the specified duration
    p.join(timeout=duration_limit_sec)

    if p.is_alive():
        print(f"Warning: Computation exceeded {duration_limit_sec}s. Terminating process...")
        p.terminate()  # Force kill the process
        p.join()       # Clean up the zombie process
        return fallback_return_value

    if not q.empty():
        result = q.get()
        if isinstance(result, Exception):
            print(f"An error occurred in worker: {result}")
            return fallback_return_value
        return result
    
    return fallback_return_value