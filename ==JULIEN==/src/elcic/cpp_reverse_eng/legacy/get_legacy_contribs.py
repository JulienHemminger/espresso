import sys
import re
import os
import tempfile
from elc.energy.legacy_elc_energy import get_legacy_energy
import numpy as np
import math


def get_legacy_contribs(system, params) -> dict:
    def _get_legacy_energy_with_capture(system, params):
        # Create a temporary file to safely capture large outputs without pipe deadlock
        with tempfile.TemporaryFile(mode="w+b") as tmp_file:
            # Duplicate current stdout and stderr for later restoration
            original_stdout_fd = os.dup(sys.stdout.fileno())
            original_stderr_fd = os.dup(sys.stderr.fileno())

            try:
                # Redirect stdout and stderr to the temp file
                os.dup2(tmp_file.fileno(), sys.stdout.fileno())
                os.dup2(tmp_file.fileno(), sys.stderr.fileno())

                # Execute the legacy code
                result = get_legacy_energy(system, params)

                # Manually flush python stdout before closing/restoring
                sys.stdout.flush()
                sys.stderr.flush()

            finally:
                # Restore original streams safely
                os.dup2(original_stdout_fd, sys.stdout.fileno())
                os.dup2(original_stderr_fd, sys.stderr.fileno())

                os.close(original_stdout_fd)
                os.close(original_stderr_fd)

            # Rewind and read the content captured in the temporary file
            tmp_file.seek(0)
            captured_output = tmp_file.read()

        return result, captured_output.decode("utf-8")

    # Call the function
    energy_dict, output = _get_legacy_energy_with_capture(system, params)

    # Parsing logic
    def _parse_elc_output(log_text):
        data = {}
        # Matches lines like: [ELC] E_near_L0_L0 = -0.27256698373464
        pattern = (
            r"\[ELC\]\s*(?P<key>\w+)\s*=\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )

        matches = re.finditer(pattern, log_text)
        for match in matches:
            key = match.group("key")
            val = float(match.group("value"))
            data[key] = val

        return data

    legacy_contribs = _parse_elc_output(output)

    # Ensure our required parsed components are present before assertion
    if (
        "E_near" in legacy_contribs
        and "E_far" in legacy_contribs
        and "E_total" in legacy_contribs
    ):
        assert math.isclose(
            legacy_contribs["E_total"],
            legacy_contribs["E_near"] + legacy_contribs["E_far"],
            rel_tol=1e-9,
        )
    else:
        raise ValueError(
            "Could not parse all required ELC energy contributions from the output."
        )

    # Your original return dict schema (with E_total matched to the actual returned float)
    legacy_contribs["E_total"] = energy_dict["total"]

    # DEBUG
    legacy_contribs["E_near"] += energy_dict[("coulomb", 0)]

    return legacy_contribs


