import random

def get_rdm_charges_neutral(charge_count=2, min_q=-10.0, max_q=10.0):
    """
    Returns a list of charges that sum to 0.0, with each charge in [min_q, max_q].
    """
    assert charge_count >= 2
    
    charges = [0.0] * charge_count
    current_sum = 0.0
    
    # We generate n-1 charges
    for i in range(charge_count - 1):
        # Determine constraints for this charge so the final charge 
        # can still potentially fall within [min_q, max_q]
        remaining_count = (charge_count - 1) - i
        
        # New range constraint:
        # The sum of remaining charges (S_rem) must be in [min_q, max_q].
        # Since S_rem = -(current_sum + q_i), this provides the bounds.
        low = max(min_q, -(current_sum + (remaining_count * max_q)))
        high = min(max_q, -(current_sum + (remaining_count * min_q)))
        
        q = random.uniform(low, high)
        charges[i] = q
        current_sum += q
        
    # Final charge must balance the sum to 0
    charges[-1] = -current_sum
    
    return charges