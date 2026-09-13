"""Normalize logging-only draft metadata scalars, preserving tensor offsets."""
import copy

def graph_metadata(value):
    if isinstance(value, dict):
        return {k: graph_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [graph_metadata(v) for v in value]
    if type(value).__name__ == 'AscendDSAMetadata':
        result = copy.copy(value)
        # In the pinned DSACP forward, only num_prefills > 0 is consumed.
        # Exact counts were used by the builder already; tensor offsets and
        # req.num_reqs_actual remain untouched. Do not specialize graph bodies
        # on logging-only decode counts or the number of prefill requests.
        result.num_prefills = int(value.num_prefills > 0)
        result.num_decodes = result.num_decode_tokens = 0
        return result
    return value
