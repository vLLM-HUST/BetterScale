"""Serving request rows and retained State seats are independent capacities."""


def seat_counts(config):
    execution = config.scheduler_config.max_num_seqs
    resident = getattr(config, "additional_config", {}).get(
        "state_resident_seats", execution + 4
    )
    if type(execution) is not int or not 1 <= execution <= 36:
        raise ValueError("Qwen35 execution seats must be in 1..36")
    if type(resident) is not int or resident < execution:
        raise ValueError("resident seats must cover the execution envelope")
    return execution, resident
