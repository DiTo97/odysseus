import numpy as np
import typing as t


def get_zone_id_from_action(action, no_op_zone_id, neighbors_dict):
    """
    Match action to zone Id, given the NOP zone id for that action ant its neighbors.

    Parameters
    ----------
    action
    no_op_zone_id
    neighbors_dict

    Returns
    -------

    """

    if action == 0:
        zone_id = no_op_zone_id
    else:
        zone_id = neighbors_dict[no_op_zone_id][action]

    if np.isnan(zone_id):
        return None

    return int(zone_id)


def omega(delta: int, d: int, n_a: int):
    """
    Omega function
    Parameters
    ----------
    delta: int
        Delta(z,t)
    d: int
        demand(z,t+T)
    n_a: int
        number of available vehicles

    Returns
    -------

    """
    if delta >= 0:
        return delta * np.exp(-1 / d * n_a)
    else:
        return delta * np.exp(-1 / max(n_a, 1) * max(d, 1))


def psi(n_d: int, d: int, n_a: int):
    """
    Psi function
    Parameters
    ----------
    n_d: int
        Number of dead vehicles at zone z at time t
    d: int
        demand(z, t+T)
    n_a: int
        number of available vehicles
    Returns
    -------

    """
    return n_d * np.exp(-1/d * n_a)


def get_demand(bookings_df, zone_id, date, hour):
    d = bookings_df[(bookings_df['origin_id'] == zone_id) &
                    (bookings_df['date'] == date) &
                    (bookings_df['hour'] == hour)].shape[0]

    # fix division by zero problem
    return d if d > 0 else 1


def get_delta(deltas_dict, zone_id):
    return int(deltas_dict[zone_id])


def choose_hour(booking_request: t.Dict, city: str):
    """
    Pick the current hour Id or the next one, based on the scheduled start time of
    the booking request: If `start_time` > `hour_30`, pick the next hour Id,
    where `hour_30` is the 30 min mark of the scheduled hour.

    Parameters
    ----------
    booking_request : t.Dict
        Booking request data

    city : str
        City undergoing the simulation

    Returns
    -------
    hour: int
        Hour in the range [0, 23]
    """

    if city == 'Austin':
        hour_0 = 18
    elif city == 'Louisville':
        hour_0 = 23
    else:
        raise ValueError('City unsupported')

    minute_0 = 45 # Fixed for any city

    hour = booking_request['hour']  # Hour Id in [0-23]

    # Hour, min from timestamp
    real_hour = booking_request['start_time'].hour
    real_min  = booking_request['start_time'].minute

    # Get the starting real hour corresponding to
    # the hour Id interval of the booking request
    # (e.g. booking_request['hour'] = 0 -> 18
    start_real_hour = (hour + hour_0) % 24

    # If you are over the first half,
    # return the next hour Id
    if (real_hour > start_real_hour) \
            and (real_min > (minute_0 + 30) % 60):
        hour = (hour + 1) % 24

    return hour


def get_dead_vehicles(chargingStrategy, zone_id):
    return len(chargingStrategy.get_dead_vehicles_by_zone(zone_id))


def compute_pick_up_reward(bookings_df, deltas_dict, available_vehicles_dict, zone_id, date, hour, pick_up_zone):
    """
    Compute reward for the pick-up agent
    Parameters
    ----------
    bookings_df: pd.DataFrame
    deltas_dict: dict
    available_vehicles_dict: dict
    zone_id: int
    date: datetime.date
    hour: int
    pick_up_zone: int

    Returns
    -------

    """

    if zone_id is not None:
        delta = get_delta(deltas_dict, zone_id)
        d = get_demand(bookings_df, zone_id, date, hour)

        no_vehicles = False

        # Get # of available vehicles by zone Id
        if zone_id in available_vehicles_dict:
            n_a = len(available_vehicles_dict[zone_id])
        else:
            n_a = 0
            no_vehicles = True

        reward = omega(delta, d, n_a)

        # -omega if action = pick-up (by definition)
        if zone_id == pick_up_zone:
            reward = -1*reward
        # if action = no pick-up and the reward is negative with no available
        # vehicles, then make the reward positive
        elif (no_vehicles is True) & (reward < 0):
            reward = -1*reward
    else:
        reward = np.nan

    return reward


def compute_drop_off_reward(bookings_df, deltas_dict, available_vehicles_dict, chargingStrategy, pick_up_vehicle,
                            driving_distance, alpha, zone_id, date, hour, final_drop_off_zone):

    if zone_id is not None:
        found_vehicle = pick_up_vehicle is not None

        delta = get_delta(deltas_dict, zone_id)
        d = get_demand(bookings_df, zone_id, date, hour)

        # Get # of available vehicles by zone Id
        if zone_id in available_vehicles_dict:
            n_a = len(available_vehicles_dict[zone_id])
        else:
            n_a = 0

        if found_vehicle:
            fuel_consumed = pick_up_vehicle.distance_to_consumption(driving_distance / 1000)
            percentage = pick_up_vehicle.consumption_to_percentage(fuel_consumed)

            final_soc = pick_up_vehicle.soc.level - percentage

            # Check if the found vehicle would have
            # working SoC battery after the trip
            if final_soc > alpha:
                reward = omega(delta, d, n_a)
            else:
                n_d = get_dead_vehicles(chargingStrategy, zone_id)
                reward = psi(n_d, d, n_a)
        else:
            reward = omega(delta, d, n_a)

        if zone_id != final_drop_off_zone \
                or not found_vehicle:
            reward = -1 * reward
    else:
        reward = np.nan

    return reward
