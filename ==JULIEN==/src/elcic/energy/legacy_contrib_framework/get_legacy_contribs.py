import sys
import re
import os
from elc.energy.legacy_elc_energy import get_legacy_energy
import numpy as np
import math

def get_legacy_contribs(system, params) -> dict:
    def _get_legacy_energy_with_capture(system, params):
        # Setup the pipe
        read_fd, write_fd = os.pipe()
        
        # Duplicate current stdout and stderr for later restoration
        original_stdout_fd = os.dup(sys.stdout.fileno())
        original_stderr_fd = os.dup(sys.stderr.fileno())
        
        try:
            # Redirect stdout and stderr to the write end of our pipe
            os.dup2(write_fd, sys.stdout.fileno())
            os.dup2(write_fd, sys.stderr.fileno())
            
            # Execute the legacy code
            # We use a try-finally to ensure we restore the streams even if this crashes
            result = get_legacy_energy(system, params)
            
            # Manually flush python stdout before closing/restoring
            sys.stdout.flush()
            
        finally:
            # Restore original streams
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            
            # Close the write end of the pipe, then read the content
            os.close(write_fd)
            
            # Read from the pipe
            # We use a blocking read or a loop to ensure we catch all output
            captured_output = b""
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                captured_output += chunk
                
            os.close(read_fd)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
            
        return result, captured_output.decode('utf-8')

    # Call the function
    total_energy, output = _get_legacy_energy_with_capture(system, params)

    # Parsing logic
    def _parse_elc_output(log_text):
        data = {}
        # Matches lines like: [ELC] E_near_L0_L0 = -0.27256698373464
        # Supports optional signs, decimals, and scientific notation
        pattern = r"\[ELC\]\s*(?P<key>\w+)\s*=\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        
        matches = re.finditer(pattern, log_text)
        for match in matches:
            key = match.group('key')
            val = float(match.group('value'))
            data[key] = val
            
        return data

    legacy_contribs = _parse_elc_output(output)
    
    # Ensure our required parsed components are present before assertion
    if 'E_near' in legacy_contribs and 'E_far' in legacy_contribs and 'E_total' in legacy_contribs:
        assert math.isclose(legacy_contribs['E_total'], legacy_contribs['E_near'] + legacy_contribs['E_far'], rel_tol=1e-9)
    else:
        raise ValueError("Could not parse all required ELC energy contributions from the output.")

    # Your original return dict schema (with E_total matched to the actual returned float)
    legacy_contribs['E_total'] = total_energy
    return legacy_contribs