import typing as t


def compute_delta(avail_vehicles_by_zone: t.Dict[int, t.List],
                  pred_out_flows: t.Dict[str,
                      t.Dict[int,
                      t.Dict[int, float]]],
                  pred_in_flows: t.Dict[str,
                      t.Dict[int,
                      t.Dict[int, float]]],
                  daytype: str, hour: int) \
                 -> t.Dict[int, float]:
    """
    Compute the Delta function for all the zones for a given daytype for a given hour, as described in
    L. Tolomei, et al. - Benefits of Relocation on E-scooter Sharing - a Data-informed Approach

    Parameters
    ----------
    avail_vehicles_by_zone : t.Dict[int, t.List]
        Available vehicles per zone Id

    pred_out_flows : t.Dict[str, t.Dict[int, t.Dict[int, float]]]
        Estimated output flow per daytype per hour per Zone Id

    pred_in_flows : t.Dict[str, t.Dict[int, t.Dict[int, float]]]
        Estimated input flow per daytype per hour per Zone Id

    daytype : str
        Daytype of choice in ['weekday', 'weekend']

    hour : int
        Hour of choice in [0, 23]

    Returns
    -------
    delta_by_zone : t.Dict[int, float]
        Estimated delta per zone Id
    """
    _h = hour % 24

    pred_out_flows_dh = pred_out_flows[daytype][_h]
    pred_in_flows_dh  = pred_in_flows[daytype][_h]

    delta_by_zone = {}

    for zone, vehicles in avail_vehicles_by_zone.items():
        pred_flow = pred_out_flows_dh[zone] \
                  - pred_in_flows_dh[zone]

        delta = pred_flow - len(vehicles)
        delta_by_zone[zone] = delta

    return delta_by_zone
