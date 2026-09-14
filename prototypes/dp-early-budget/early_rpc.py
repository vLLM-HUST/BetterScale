"""Native MP callable seam: preserve wrapper cache handling and pass the budget."""


def execute_with_budget(wrapper, scheduler_output, early_budget):
    # WorkerWrapperBase.execute_model has a fixed one-argument signature.
    # Preserve its cache application before entering our worker, rather than
    # adding a keyword to that native wrapper or bypassing cache handling.
    wrapper._apply_mm_cache(scheduler_output)
    return wrapper.worker.execute_model(scheduler_output, early_budget=early_budget)
