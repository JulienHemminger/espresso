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
        pattern = r"E_far_p3m\s*=\s*(?P<p3m>[-+]?\d*\.\d+),\s*E_far_corr\s*=\s*(?P<corr>[-+]?\d*\.\d+),\s*E_far\s*=\s*.*=\s*(?P<far>[-+]?\d*\.\d+)"
        
        match = re.search(pattern, log_text)
        if match:
            data.update({
                'E_near_p3m': float(match.group('p3m')),
                'E_near_corr': float(match.group('corr')),
                'E_near': float(match.group('far'))
            })
        return data

    legacy_contribs = _parse_elc_output(output)
    assert math.isclose(legacy_contribs['E_near'], legacy_contribs['E_near_p3m']+legacy_contribs['E_near_corr'])

    legacy_contribs['E_total'] = total_energy
    legacy_contribs['E_far'] = legacy_contribs['E_total'] - legacy_contribs['E_near']
    return legacy_contribs