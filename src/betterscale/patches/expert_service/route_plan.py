"""CPU-only wire admission for optional native client route preparation."""


def threshold(abi):
    mode = abi.get('client_route_plan')
    if mode is None:
        if 'client_route_plan_min_rows' in abi:
            raise ValueError('Route-plan threshold without a wire mode')
        return 0
    rows = abi.get('client_route_plan_min_rows')
    if (mode != 'native-v2-cap1' or type(rows) is not int or not 1 <= rows <= 4096
            or abi.get('placement') != 'layer' or abi.get('sources_per_wave') != 1):
        raise ValueError('Native route-plan ABI requires layer/cap1 and threshold1..4096')
    return rows
