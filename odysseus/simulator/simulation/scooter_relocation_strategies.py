import itertools
import os
import sys
from collections import deque
from queue import PriorityQueue, Empty

import six

sys.modules['sklearn.externals.six'] = six
from mlrose import TSPOpt, genetic_alg

import numpy as np
import pandas as pd
from odysseus.simulator.simulation.scooter_relocation_primitives import *
from odysseus.utils.time_utils import weekday2vec


class ScooterRelocationStrategy(ScooterRelocationPrimitives):

    def check_scooter_relocation(self, booking_request, vehicles=None):
        """

        Parameters
        ----------
        booking_request
        vehicles

        Returns
        -------

        """
        relocated = False
        relocated_vehicles = vehicles
        scooter_relocation = {}

        if self.simInput.supply_model_conf["battery_swap"]:

            if self.simInput.supply_model_conf["scooter_relocation_strategy"] == "magic_relocation":

                booking_request_zone_column = int(np.floor(
                    booking_request["origin_id"] / self.simInput.grid_matrix.shape[0]
                ))
                booking_request_zone_row = int(
                    booking_request["origin_id"] - booking_request_zone_column * self.simInput.grid_matrix.shape[0]
                )

                found_vehicle_flag = False
                r = 2  # excludes origin zone and its neighbors

                while not found_vehicle_flag and r < max(self.simInput.grid_matrix.shape):

                    zones_ring = []
                    available_vehicles_soc_dict = {}

                    i = booking_request_zone_row - r
                    j = booking_request_zone_column - r
                    if i >= 0 and j >= 0:
                        while j < booking_request_zone_column + r and j < self.simInput.grid_matrix.shape[1]:
                            zones_ring.append(self.simInput.grid_matrix.iloc[i, j])
                            j += 1

                    i = booking_request_zone_row - r
                    j = booking_request_zone_column + r
                    if i >= 0 and j < self.simInput.grid_matrix.shape[1]:
                        while i < booking_request_zone_row + r and i < self.simInput.grid_matrix.shape[0]:
                            zones_ring.append(self.simInput.grid_matrix.iloc[i, j])
                            i += 1

                    i = booking_request_zone_row + r
                    j = booking_request_zone_column + r
                    if i < self.simInput.grid_matrix.shape[0] and j < self.simInput.grid_matrix.shape[1]:
                        while j > booking_request_zone_column - r and j > 0:
                            zones_ring.append(self.simInput.grid_matrix.iloc[i, j])
                            j -= 1

                    i = booking_request_zone_row + r
                    j = booking_request_zone_column - r
                    if i < self.simInput.grid_matrix.shape[0] and j >= 0:
                        while i > booking_request_zone_row - r and i > 0:
                            zones_ring.append(self.simInput.grid_matrix.iloc[i, j])
                            i -= 1

                    for zone_id in zones_ring:
                        if zone_id in self.available_vehicles_dict and len(self.available_vehicles_dict[zone_id]):
                            for available_vehicle in self.available_vehicles_dict[zone_id]:
                                available_vehicles_soc_dict[available_vehicle] = self.vehicles_list[available_vehicle].soc.level

                    if len(available_vehicles_soc_dict):
                        max_soc_vehicle = max(available_vehicles_soc_dict, key=available_vehicles_soc_dict.get)
                        max_soc_vehicle_zone = self.vehicles_zones[max_soc_vehicle]

                        if self.vehicles_list[max_soc_vehicle].soc.level > abs(
                            self.vehicles_list[max_soc_vehicle].consumption_to_percentage(
                                self.vehicles_list[max_soc_vehicle].distance_to_consumption(
                                    booking_request["driving_distance"] / 1000
                                )
                            )
                        ):
                            found_vehicle_flag = True

                    r += 1

                if found_vehicle_flag:

                    relocated = True
                    relocation_zone_id = booking_request["origin_id"]
                    relocated_vehicles = [max_soc_vehicle]

                    distance = get_od_distance(
                        self.simInput.grid,
                        max_soc_vehicle_zone,
                        relocation_zone_id
                    )

                    scooter_relocation = init_scooter_relocation(relocated_vehicles, booking_request["start_time"],
                                                                 [max_soc_vehicle_zone], [relocation_zone_id],
                                                                 distance, 0)

            else:

                relocation_zone_id = None

                if self.simInput.supply_model_conf["scooter_relocation_strategy"] in ["reactive_post_charge",
                                                                                      "reactive_post_trip"]:

                    scheduled_relocation = None
                    for proposed_relocation in self.scheduled_scooter_relocations:
                        if booking_request["destination_id"] in proposed_relocation["pick_up"]:
                            scheduled_relocation = proposed_relocation

                    if scheduled_relocation:
                        self.scheduled_scooter_relocations.remove(scheduled_relocation)

                        relocation_zone_id, n_relocated_vehicles = scheduled_relocation["drop_off"].popitem()

                        if relocated_vehicles is not None:
                            for i in range(
                                    len(relocated_vehicles),  # first vehicles are input vehicles
                                    min(
                                        n_relocated_vehicles,
                                        len(self.available_vehicles_dict[booking_request["destination_id"]])
                                    )):
                                relocated_vehicle = self.available_vehicles_dict[booking_request["destination_id"]].pop()
                                relocated_vehicles.append(relocated_vehicle)
                        else:
                            relocated_vehicles = []
                            for i in range(
                                    0,
                                    min(
                                        n_relocated_vehicles,
                                        len(self.available_vehicles_dict[booking_request["destination_id"]])
                                    )):
                                relocated_vehicle = self.available_vehicles_dict[booking_request["destination_id"]].pop()
                                relocated_vehicles.append(relocated_vehicle)

                else:
                    relocation_zone_ids, _ = self.choose_ending_zone(daytype=booking_request["daytype"],
                                                                     hour=booking_request["hour"])
                    relocation_zone_id = relocation_zone_ids[0]

                if relocation_zone_id and relocation_zone_id != booking_request["destination_id"] \
                        and relocated_vehicles:

                    relocated = True

                    distance = get_od_distance(
                        self.simInput.grid,
                        booking_request["destination_id"],
                        relocation_zone_id
                    )

                    duration = distance / 1000 / self.simInput.supply_model_conf["avg_relocation_speed"] * 3600

                    scooter_relocation = init_scooter_relocation(relocated_vehicles, booking_request["end_time"],
                                                                 [booking_request["destination_id"]],
                                                                 [relocation_zone_id],
                                                                 distance, duration)

        return relocated, scooter_relocation

    def choose_ending_zone(self, n=1, origin_scores_list=None, destination_scores_list=None, daytype=None, hour=None):
        """
        Chooses n ending zones given a list of origin and destination scores, according to the selected zone selection
        technique. If the technique defines a priority (e.g.: vehicles aggregation or Delta value), returned lists are
        ordered by relocation priority.
        :param n: Maximum number of zones to be selected as ending zones.
        :param origin_scores_list: List of origin scores. It is required by 'Delta' technique. Each element of this list
        should correspond to a dictionary of probabilities for a given hour. Each value of this dictionary should be the
        probability for a given zone to be selected as an origin zone for a trip.
        :param destination_scores_list: List of destination scores. It is required by 'Delta' technique. Its structure
        is similar to origin_scores_list. See the latter for further details.
        :param daytype: Current simulated type of day (i.e.: 'weekday', 'weekend'). Required by 'kde_sampling' technique.
        :param hour: Current simulated hour. Required by 'kde_sampling' technique.
        :return: A list of proposed ending zones and a list of proposed numbers of vehicles to be dropped off into such
        zones. If the technique defines a priority, returned lists are ordered by relocation priority.
        """
        ending_zone_ids = []
        n_dropped_vehicles_list = []
        technique = dict(self.simInput.supply_model_conf["scooter_relocation_technique"])["end"]

        if technique == "kde_sampling":

            next_hour_kde = self.simInput.trip_kdes[daytype][(hour + 1) % 24]

            def base_round(x, base):
                if x < 0:
                    return 0
                elif x > base:
                    return base
                else:
                    return round(x)

            def gen_relocation_zone(kde):
                trip_sample = kde.sample()
                origin_i = base_round(trip_sample[0][0], len(self.simInput.grid_matrix.index) - 1)
                origin_j = base_round(trip_sample[0][1], len(self.simInput.grid_matrix.columns) - 1)

                return self.simInput.grid_matrix.loc[origin_i, origin_j]

            for i in range(n):
                origin_id = gen_relocation_zone(next_hour_kde)
                while (origin_id not in self.simInput.valid_zones) or (origin_id in self.starting_zone_ids):
                    origin_id = gen_relocation_zone(next_hour_kde)
                ending_zone_ids.append(origin_id)
                n_dropped_vehicles_list.append(1)

        if technique == "aggregation":

            n_vehicles_by_zone = {
                k: len(v) for k, v in
                sorted(self.available_vehicles_dict.items(), key=lambda item: -len(item[1]))
            }

            for i in range(min(n, len(n_vehicles_by_zone))):
                ending_zone_ids.append(n_vehicles_by_zone.popitem()[0])
                n_dropped_vehicles_list.append(1)

        if technique == "delta":

            delta_by_zone = self.compute_delta(origin_scores_list, destination_scores_list)

            delta_by_zone = {
                zone: delta for zone, delta in
                sorted(delta_by_zone.items(), key=lambda item: item[1])
                if delta > 0
            }

            for i in range(min(n, len(delta_by_zone))):
                zone, delta = delta_by_zone.popitem()
                n_dropped_vehicles = int(delta)
                ending_zone_ids.append(zone)
                n_dropped_vehicles_list.append(n_dropped_vehicles)

        return ending_zone_ids, n_dropped_vehicles_list

    def choose_starting_zone(self, n=1, pred_out_flows_list=None, pred_in_flows_list=None):
        """
        Chooses n starting zones given a list of origin and destination scores, according to the selected zone selection
        technique. If the technique defines a priority (e.g.: vehicles aggregation or Delta value), returned lists are
        ordered by relocation priority.

        :param n: Maximum number of zones to be selected as starting zones.
        :param pred_out_flows_list: List of hourly predicted out flows. It is required by 'Delta' technique. Each
        element of this list should correspond to a dictionary of predicted flows for a given hour. Each value of this
        dictionary should be the predicted out flow for a given zone.
        :param pred_in_flows_list: List of hourly predicted in flows. It is required by 'Delta' technique. Its structure
        is similar to out_flows_list. See the latter for further details.
        :return: A list of proposed starting zones and a list of proposed numbers of vehicles to be picked up from such
        zones. If the technique defines a priority, returned lists are ordered by relocation priority.
        """
        starting_zone_ids = []
        n_picked_vehicles_list = []
        technique = dict(self.simInput.supply_model_conf["scooter_relocation_technique"])["start"]

        if technique == "aggregation":

            n_vehicles_by_zone = {
                k: len(v) for k, v in
                sorted(self.available_vehicles_dict.items(), key=lambda item: len(item[1]))
            }

            for i in range(min(n, len(n_vehicles_by_zone))):
                starting_zone_ids.append(n_vehicles_by_zone.popitem()[0])
                n_picked_vehicles_list.append(1)

        if technique == "delta":

            delta_by_zone = self.compute_delta(pred_out_flows_list, pred_in_flows_list)

            # Codice da oscar per scartare zone con delta positivi e ordinare i delta negativi in ordine discendente
            # (dal numero negativo più grande al numero negativo più piccolo, es. -3, ... , -100)

            # per primo vale if delta < 0 -> mantiene nel dizionario solo i delta negativi (surplus of vehicles -> pick-up zones)
            # quindi sui delta negativi chiama sorted sulla lista di tuple delta_by_zone.items()
            # sorted è di default in ordine ascendente, quindi è necessario specificare ordine discendente oppure come
            # chiave del confronto invertire il segno di delta, cioè quello che viene fatto con key=....)
            delta_by_zone = {
                zone: delta for zone, delta in
                sorted(delta_by_zone.items(), key=lambda item: -item[1]) if delta < 0
            }

            # Crea due liste in cui riporta le top-priority zones ed i rispettivi delta (resi positivi)
            for i in range(min(n, len(delta_by_zone))):
                zone, delta = delta_by_zone.popitem() # popitem() prende dalla fine del dizionario, quindi le zone con delta più basso
                n_picked_vehicles = int(-delta) # delta è negativo
                starting_zone_ids.append(zone)
                n_picked_vehicles_list.append(n_picked_vehicles)

        return starting_zone_ids, n_picked_vehicles_list

    def generate_relocation_schedule(self, current_datetime, daytype, hour):
        """
        It generates a relocation schedule for a given hour of a day of a given type (i.e.: 'weekday', 'weekend').

        If the relocation strategy requires it, once generation is finished, the relocation process is automatically
        triggered.

        :param current_datetime: current time of simpy simulator
        :param daytype: Current simulated type of day (i.e.: 'weekday', 'weekend')
        :param hour: Current simulated hour.
        :return: void
        """

        # List containing a dictionary with the relocations to be performed:
        #   {'pick_up' : {'zone_id', tot_relocated_vehicles},
        #    'drop_off' : {'zone_id', tot_relocated_vehicles}}
        # Each relocation is completed by a worker
        self.scheduled_scooter_relocations.clear()

        pred_out_flows_list = []
        pred_in_flows_list = []

        if self.simInput.supply_model_conf["scooter_relocation_strategy"] == "predictive":

            # Prepare flows from past hours
            if self.current_hour_origin_count:
                self.past_hours_origin_counts.append(self.current_hour_origin_count)
                self.past_hours_destination_counts.append(self.current_hour_destination_count)

                past_in_flow = np.zeros((2, self.city_shape[0], self.city_shape[1]))
                past_out_flow = np.zeros((2, self.city_shape[0], self.city_shape[1]))

                if len(self.past_hours_origin_counts) > 2:
                    self.past_hours_origin_counts.pop(0)
                    self.past_hours_destination_counts.pop(0)

                if len(self.past_hours_origin_counts) >= 2:
                    for i in range(len(self.past_hours_origin_counts)):
                        past_origin_count = self.past_hours_origin_counts[i]
                        past_destination_count = self.past_hours_destination_counts[i]

                        for j in self.simInput.grid_matrix.index:
                            for k in self.simInput.grid_matrix.columns:
                                zone = self.simInput.grid_matrix.iloc[j, k]

                                if zone in self.simInput.valid_zones:
                                    if zone in past_origin_count:
                                        past_in_flow[i][j][k] = past_origin_count[zone]
                                    if zone in past_destination_count:
                                        past_out_flow[i][j][k] = past_destination_count[zone]

                    max_flow = max(self.simInput.max_out_flow, self.simInput.max_in_flow)
                    prediction_datetime = current_datetime
                    prediction_weekday = prediction_datetime.weekday()

                    # Tensor creation
                    d1 = np.concatenate([
                        np.expand_dims(past_in_flow, axis=0),
                        np.expand_dims(past_out_flow, axis=0)
                    ], axis=0)

                    # Flux dimension displaced at the end of the tensor
                    d1 = np.moveaxis(d1, 0, -1)

                    # Adding a dimension
                    d1 = d1[np.newaxis, ...]

                    X_test = [d1]

                    meta_data = True
                    holiday_data = False
                    meteorol_data = False

                    # Adding metadata
                    meta_feature = []
                    if meta_data:
                        time_feature = weekday2vec([prediction_weekday])
                        meta_feature.append(time_feature)
                        if holiday_data:
                            pass
                            # load holiday
                            #holiday_feature = load_holiday(timestamps_Y, datapath)
                            #meta_feature.append(holiday_feature)
                        if meteorol_data:
                            pass
                            # load meteorol data
                            #meteorol_feature = load_meteorol(timestamps_Y, datapath)
                            #meta_feature.append(meteorol_feature)

                        meta_feature = np.hstack(meta_feature) if len(meta_feature) > 0 else np.asarray(meta_feature)
                        X_test.append(meta_feature)

                    prediction = self.prediction_model_time_horizon_two.predict(X_test, max_flow)

                    pred_in_flows = {}
                    pred_out_flows = {}
                    for j in self.simInput.grid_matrix.index:
                        for k in self.simInput.grid_matrix.columns:
                            zone = self.simInput.grid_matrix.iloc[j, k]
                            pred_in_flows[zone] = prediction[0][j][k][0]
                            pred_out_flows[zone] = prediction[0][j][k][1]

                    pred_in_flows_list.append(pred_in_flows)
                    pred_out_flows_list.append(pred_out_flows)

                    if self.window_width > 1:

                        prediction = self.prediction_model_time_horizon_three.predict(X_test, max_flow)

                        pred_in_flows = {}
                        pred_out_flows = {}
                        for j in self.simInput.grid_matrix.index:
                            for k in self.simInput.grid_matrix.columns:
                                zone = self.simInput.grid_matrix.iloc[j, k]
                                pred_in_flows[zone] = prediction[0][j][k][0]
                                pred_out_flows[zone] = prediction[0][j][k][1]

                        pred_in_flows_list.append(pred_in_flows)
                        pred_out_flows_list.append(pred_out_flows)

            else:
                return

            self.reset_current_hour_stats()

        else:
            
            # BASELINE STRATEGY
            
            # Get avg flows from past data (precomputed with demand_modelling)
            pred_out_flows = self.simInput.avg_out_flows_train
            pred_in_flows = self.simInput.avg_in_flows_train

            for i in range(self.window_width): # window_width è di default 1 quindi i = 0 e semplicemente
                # carica il dizionario delle predizioni dei flussi per quel daytype, per quell'ora in una lista
                # FIXME: Inutile utilizzare una lista qui
                pred_out_flows_list.append(pred_out_flows[daytype][(hour + 1 + i) % 24])
                pred_in_flows_list.append(pred_in_flows[daytype][(hour + 1 + i) % 24])

        if pred_out_flows_list and pred_in_flows_list:

            # Choose the maximum number of 'pick up' and 'drop off' zones proposals to be computed
            n_relocations = int(len(self.available_vehicles_dict) / 2)  # an upper bound

            if self.simInput.supply_model_conf["scooter_relocation_strategy"] in ["proactive", "predictive"] \
                    and "relocation_capacity" not in self.simInput.supply_model_conf:
                n_free_workers = self.relocation_workers_resource.capacity - self.relocation_workers_resource.count
                n_relocations = min(n_relocations, n_free_workers)

            # Compute proposals of starting and ending zones for relocations
            # FIXME: Use the number of workers as upper bound instead of n_relocations
            self.starting_zone_ids, self.n_picked_vehicles_list = self.choose_starting_zone(n=n_relocations,
                                                                                            pred_out_flows_list=pred_out_flows_list,
                                                                                            pred_in_flows_list=pred_in_flows_list)


            self.ending_zone_ids, self.n_dropped_vehicles_list = self.choose_ending_zone(n=n_relocations,
                                                                                         origin_scores_list=pred_out_flows_list,
                                                                                         destination_scores_list=pred_in_flows_list,
                                                                                         daytype=daytype, hour=hour)

            if self.starting_zone_ids and self.ending_zone_ids:

                if self.simInput.supply_model_conf["scooter_relocation_strategy"] in ["proactive", "predictive"] \
                        and "relocation_capacity" in self.simInput.supply_model_conf:
                    # Distribute proposed 'pick up' and 'drop off' zones between scheduled relocations

                    relocation_capacity = self.simInput.supply_model_conf["relocation_capacity"]

                    pick_up_zone_ids = self.starting_zone_ids.copy()
                    n_picked_vehicles_list = self.n_picked_vehicles_list.copy()
                    drop_off_zone_ids = self.ending_zone_ids.copy()
                    n_dropped_vehicles_list = self.n_dropped_vehicles_list.copy()

                    first_pick_up_zone_id = pick_up_zone_ids[0] # Least priority pick-up zone

                    # Create priority queues combining relocation priority and distance
                    pick_up_zone_priority_queue = self.compute_zone_priorities(first_pick_up_zone_id, pick_up_zone_ids[1:])
                    drop_off_zones_priority_queue = self.compute_zone_priorities(first_pick_up_zone_id, drop_off_zone_ids)

                    pick_up_zone_index = 0
                    try:
                        priority, drop_off_zone_index = drop_off_zones_priority_queue.get_nowait() # get top-priority drop_off zone
                    except Empty:
                        return

                    satisfied_drop_off_zones_indexes = []
                    empty_queue = False

                    for i in range(n_relocations):
                        if empty_queue:
                            break

                        residual_capacity = relocation_capacity
                        scheduled_relocation = {
                            "pick_up": {},
                            "drop_off": {}
                        }

                        if "profitability_check" in self.simInput.supply_model_conf:
                            relocation_profitability_check = self.simInput.supply_model_conf["profitability_check"]
                        else:
                            relocation_profitability_check = True

                        if "worker_truck_consumption" in self.simInput.supply_model_conf:
                            relocation_vehicle_consumption = self.simInput.supply_model_conf["worker_truck_consumption"]
                        else:
                            relocation_vehicle_consumption = 7  # l/100km

                        if "diesel_price" in self.simInput.supply_model_conf:
                            diesel_price = self.simInput.supply_model_conf["diesel_price"]
                        else:
                            diesel_price = 0.65  # $/l (USA)

                        if "unlock_fee" in self.simInput.supply_model_conf:
                            unlock_fee = self.simInput.supply_model_conf["unlock_fee"]
                        else:
                            unlock_fee = 1  # $

                        if "rent_fee" in self.simInput.supply_model_conf:
                            rent_fee = self.simInput.supply_model_conf["rent_fee"]
                        else:
                            rent_fee = 0.15  # $/min

                        if "avg_relocation_distance" in self.simInput.supply_model_conf:
                            avg_relocation_distance = self.simInput.supply_model_conf["avg_relocation_distance"]
                        else:
                            avg_relocation_distance = 1  # km

                        if "avg_trip_duration" in self.simInput.supply_model_conf:
                            avg_trip_duration = self.simInput.supply_model_conf["avg_trip_duration"]
                        else:
                            avg_trip_duration = 10  # min

                        relocation_vehicle_cost_per_km = relocation_vehicle_consumption * diesel_price

                        unitary_relocation_cost = avg_relocation_distance * relocation_vehicle_cost_per_km
                        unitary_scooter_revenue = unlock_fee + rent_fee * avg_trip_duration

                        tot_relocation_cost = 0
                        tot_potential_revenues = 0
                        n_relocated_vehicles = 0
                        was_positive = False

                        while residual_capacity > 0 and not empty_queue:
                            if relocation_profitability_check and n_relocated_vehicles:
                                if tot_potential_revenues - tot_relocation_cost >= 0:
                                    was_positive = True
                                if was_positive and tot_potential_revenues - tot_relocation_cost < 0:
                                    break

                            try:
                                pick_up_zone_id = pick_up_zone_ids[pick_up_zone_index]
                                n_picked_vehicles = n_picked_vehicles_list[pick_up_zone_index]
                                drop_off_zone_id = drop_off_zone_ids[drop_off_zone_index]
                                n_dropped_vehicles = n_dropped_vehicles_list[drop_off_zone_index]

                                if n_picked_vehicles > residual_capacity:
                                    # Try to consume the entire residual capacity
                                    if residual_capacity > n_dropped_vehicles:
                                        # 'Drop off' zone needs are satisfied before
                                        n_relocated_vehicles = n_dropped_vehicles
                                        satisfied_drop_off_zones_indexes.append(drop_off_zone_index)
                                        priority, drop_off_zone_index = drop_off_zones_priority_queue.get_nowait()

                                    elif residual_capacity < n_dropped_vehicles:
                                        # Residual capacity is totally consumed before
                                        n_relocated_vehicles = residual_capacity
                                        n_picked_vehicles_list[pick_up_zone_index] -= residual_capacity
                                        n_dropped_vehicles_list[drop_off_zone_index] -= residual_capacity

                                    else:
                                        # 'Drop off' zone needs coincide with residual capacity
                                        n_relocated_vehicles = residual_capacity
                                        n_picked_vehicles_list[pick_up_zone_index] -= residual_capacity
                                        satisfied_drop_off_zones_indexes.append(drop_off_zone_index)
                                        priority, drop_off_zone_index = drop_off_zones_priority_queue.get_nowait()

                                else:
                                    # Try to satisfy all 'pick up' zone needs
                                    if n_picked_vehicles > n_dropped_vehicles:
                                        # 'Drop off' zone needs are satisfied before 'pick up' ones
                                        n_relocated_vehicles = n_dropped_vehicles
                                        n_picked_vehicles_list[pick_up_zone_index] -= n_dropped_vehicles
                                        satisfied_drop_off_zones_indexes.append(drop_off_zone_index)
                                        priority, drop_off_zone_index = drop_off_zones_priority_queue.get_nowait()

                                    else:
                                        # 'Pick up' zone needs totally satisfied
                                        if n_picked_vehicles < n_dropped_vehicles:
                                            # Starting zone needs are satisfied before ending ones
                                            n_relocated_vehicles = n_picked_vehicles
                                            n_dropped_vehicles_list[drop_off_zone_index] -= n_picked_vehicles

                                        else:
                                            # 'Drop off' zone needs coincide with 'pick up' zone ones
                                            n_relocated_vehicles = n_picked_vehicles
                                            satisfied_drop_off_zones_indexes.append(drop_off_zone_index)

                                        priority, pick_up_zone_index = pick_up_zone_priority_queue.get_nowait()
                                        new_pick_up_zone_id = pick_up_zone_ids[pick_up_zone_index]

                                        # Re-compute 'drop off' zone priorities for new 'pick up' zone
                                        for index in sorted(satisfied_drop_off_zones_indexes, reverse=True):
                                            del drop_off_zone_ids[index]
                                            del n_dropped_vehicles_list[index]
                                        satisfied_drop_off_zones_indexes.clear()

                                        drop_off_zones_priority_queue = self.compute_zone_priorities(new_pick_up_zone_id,
                                                                                                     drop_off_zone_ids)

                                        priority, drop_off_zone_index = drop_off_zones_priority_queue.get_nowait()

                            except Empty:
                                # Ran out of 'pick up' or 'drop off' zones
                                empty_queue = True
                                break

                            finally:
                                if pick_up_zone_id not in scheduled_relocation["pick_up"]:
                                    scheduled_relocation["pick_up"][pick_up_zone_id] = n_relocated_vehicles
                                    tot_relocation_cost += unitary_relocation_cost  # added new 'pick up' zone
                                else:
                                    scheduled_relocation["pick_up"][pick_up_zone_id] += n_relocated_vehicles

                                scheduled_relocation["drop_off"][drop_off_zone_id] = n_relocated_vehicles
                                tot_relocation_cost += unitary_relocation_cost  # added new 'drop off' zone

                                residual_capacity -= n_relocated_vehicles
                                tot_potential_revenues += n_relocated_vehicles * unitary_scooter_revenue

                        # Null schedules filtering
                        scheduled_pick_up  = {
                            k: v for k, v
                                 in scheduled_relocation['pick_up'].items()
                                 if v > 0}

                        scheduled_drop_off = {
                            k: v for k, v
                                 in scheduled_relocation['drop_off'].items()
                                 if v > 0}

                        if len(scheduled_pick_up) == 0 \
                                or len(scheduled_drop_off) == 0:
                            continue

                        scheduled_relocation['pick_up']  = scheduled_pick_up
                        scheduled_relocation['drop_off'] = scheduled_drop_off

                        if relocation_profitability_check:
                            if was_positive:
                                self.scheduled_scooter_relocations.append(scheduled_relocation)
                                continue

                        self.scheduled_scooter_relocations.append(scheduled_relocation)

                else:
                    # Naive relocation strategy:
                    # Associate one 'pick up' zone and one 'drop off' zone to each scheduled relocation,
                    # following the orders of the top-priority zones

                    # FIXME: Upper bound has already been applied to create the two lists, so here n_relocations
                    # is not useful. The number of workers should be considered here
                    for i in range(min(n_relocations, len(self.starting_zone_ids), len(self.ending_zone_ids))):
                        pick_up_zone_id = self.starting_zone_ids[i]
                        drop_off_zone_id = self.ending_zone_ids[i]

                        n_picked_vehicles = self.n_picked_vehicles_list[i]
                        n_dropped_vehicles = self.n_dropped_vehicles_list[i]

                        # FIXME: Add relocation capacity here (in order to be coherent with the paper)
                        tot_relocated_vehicles = min(
                            n_picked_vehicles,
                            n_dropped_vehicles,
                        )

                        scheduled_relocation = {
                            "pick_up": {pick_up_zone_id: tot_relocated_vehicles},
                            "drop_off": {drop_off_zone_id: tot_relocated_vehicles}
                        }

                        self.scheduled_scooter_relocations.append(scheduled_relocation)

                if self.simInput.supply_model_conf["scooter_relocation_strategy"] in ["proactive", "predictive"]:
                    # Try to trigger immediately the relocation process

                    if len(self.scheduled_scooter_relocations) == 0:
                        return

                    # FIXME: Perchè n_free_workers non è len(free_workers)?
                    n_free_workers = self.relocation_workers_resource.capacity - self.relocation_workers_resource.count
                    free_workers = [worker for worker in self.relocation_workers if not worker.busy]

                    if free_workers:
                        # Compute distances between workers and the first 'pick up' zone of each relocation

                        # (key = first pick-up zone id for each relocation task, dict)
                        # The inner dict has: (key = Worker object, value = distance from that first pick-up zone-ID)
                        workers_distances_by_zone = {}
                        for (worker, scheduled_relocation) in itertools.product(free_workers, self.scheduled_scooter_relocations):
                            first_pick_up_zone_id = list(scheduled_relocation["pick_up"].keys())[0]
                            if first_pick_up_zone_id not in workers_distances_by_zone:
                                workers_distances_by_zone[first_pick_up_zone_id] = {}
                            workers_distances_by_zone[first_pick_up_zone_id][worker] = get_od_distance(
                                self.simInput.grid,
                                worker.current_position,
                                first_pick_up_zone_id
                            )

                        for i in range(min(n_relocations, len(self.scheduled_scooter_relocations), n_free_workers)):

                            scheduled_relocation = self.scheduled_scooter_relocations[i]
                            first_pick_up_zone_id = list(scheduled_relocation["pick_up"].keys())[0]

                            # Find nearest worker to the first 'pick up' zone
                            nearest_worker = None
                            nearest_worker_distance = float('inf')
                            workers_distances = workers_distances_by_zone[first_pick_up_zone_id]

                            # TODO: Select nearest worker not busy
                            for worker in workers_distances:
                                if workers_distances[worker] < nearest_worker_distance:
                                    nearest_worker = worker
                                    nearest_worker_distance = workers_distances[worker]

                            # Compute shortest path between first 'pick up' zone and the others
                            collection_path = self.compute_shortest_path(
                                first_pick_up_zone_id,
                                list(scheduled_relocation["pick_up"].keys())[1:]
                            )
                            # Compute shortest path between last 'pick up' zone and 'drop off' zones
                            distribution_path = self.compute_shortest_path(
                                collection_path[-1],
                                scheduled_relocation["drop_off"].keys()
                            )

                            # Add to collection path the step between current worker position and first 'pick up' zone
                            final_collection_path = [nearest_worker.current_position] + collection_path

                            # Trigger multi-step relocation
                            self.env.process(self.relocate_scooter_multiple_zones(scheduled_relocation,
                                                                                  final_collection_path, distribution_path,
                                                                                  nearest_worker))

    def compute_delta(self, pred_out_flows_list, pred_in_flows_list):
        """
        Compute delta of all the zones (for a given daytype and a given hour)

        Parameters
        ----------
        pred_out_flows_list: list
            for the baseline strategy it's a list containing a single element, which is the dictionary of predicted
            out flows for a given daytype, hour (having zone_ids as keys)
        pred_in_flows_list: list
            for the baseline strategy it's a list containing a single element, which is the dictionary of predicted
            in flows for a given daytype, hour (having zone_ids as keys)
        Returns
        -------
        delta_by_zone : dict
            Dictionary having zone_ids as keys, deltas as values.
        """
        window_width = len(pred_out_flows_list) # -> la lista ha solo un elemento per la baseline strategy quindi
        # window width è pari a 1

        delta_by_zone = {}
        for zone, vehicles in self.available_vehicles_dict.items():
            flow_prediction = 0
            for i in range(window_width):
                flow_prediction += pred_out_flows_list[i][zone]
                flow_prediction -= pred_in_flows_list[i][zone]
            flow_prediction /= window_width
            delta = flow_prediction - len(vehicles)
            delta_by_zone[zone] = delta

        return delta_by_zone

    def compute_shortest_path(self, starting_zone_id, other_zone_ids):
        zones = [starting_zone_id]
        [zones.append(zone) for zone in other_zone_ids]

        coords_list = []
        for zone in zones:
            zone_column = int(np.floor(
                zone / self.simInput.grid_matrix.shape[0]
            ))
            zone_row = int(
                zone - zone_column * self.simInput.grid_matrix.shape[0]
            )
            coords_list.append((zone_column, zone_row))

        problem = TSPOpt(length=len(coords_list), coords=coords_list, maximize=False)
        best_path, _ = genetic_alg(problem, max_iters=10, random_state=2)

        best_path = deque(best_path)
        starting_point = best_path[0]
        while starting_point != 0:  # Starting point is not worker position
            best_path.rotate(1)
            starting_point = best_path[0]

        return [zones[i] for i in list(best_path)]

    def compute_zone_priorities(self, starting_zone_id, other_zone_ids):
        """

        Parameters
        ----------
        starting_zone_id
        other_zone_ids

        Returns
        -------
        zones_priority_queue : PriorityQueue
            List of priorities for each zone (priority number, index of the zone in the list)
        """
        zones_priority_queue = PriorityQueue()

        # Compute distance of all the other zones w.r.t. the starting zone
        # Compute the maximum distance
        zone_distances = []
        max_distance = -1
        for i in range(len(other_zone_ids)):
            other_zone_id = other_zone_ids[i]
            distance_from_starting_zone = get_od_distance(
                self.simInput.grid,
                starting_zone_id,
                other_zone_id
            )
            zone_distances.append(distance_from_starting_zone)
            if distance_from_starting_zone > max_distance:
                max_distance = distance_from_starting_zone

        for i in range(len(other_zone_ids)):
            distance_from_starting_zone = zone_distances[i]

            priority = int((i / len(other_zone_ids))*100
                         + (distance_from_starting_zone / max_distance)*100)

            zones_priority_queue.put(
                (priority, i))

        return zones_priority_queue
