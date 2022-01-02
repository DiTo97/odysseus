import itertools as it
import numpy as np
import queue
import six
import sys

from collections import deque

# Fix mlrose dependency issues
sys.modules['sklearn.externals.six'] = six

from mlrose import TSPOpt
from mlrose import genetic_alg

# Custom imports
from odysseus.simulator.simulation.scooter_charging_primitives import ScooterChargingPrimitive


class ScooterChargingStrategy(ScooterChargingPrimitive):
    def generate_swap_schedule(self):
        """
        Generate a battery swap schedule for the current hour.
        """
        if not self.sim_input.supply_model_conf["battery_swap"]:
            raise NotImplementedError()

        self.scheduled_swaps.clear() # Clear scheduling

        n_free_workers = len([w for w in self.workers.items
                              if not w.busy])

        if n_free_workers == 0:
            return

        starting_zones_ids, n_dead_vehicles, \
                            min_zone_id = self.__pick_starting_zones()

        # Ensure there are dead vehicles
        if not starting_zones_ids      \
                or not n_dead_vehicles \
                or all(n == 0 for n in n_dead_vehicles):
            return

        battery_swap_capacity = self.sim_input.supply_model_conf['battery_swap_capacity']

        zones_priority_queue = self.__compute_zones_priority(min_zone_id,
                                                             starting_zones_ids)

        #
        # Profitability trace parameters
        #

        swap_profitability_check = self.get_supply_param(
            "profitability_check", True)

        swap_vehicle_consumption = self.get_supply_param(
            "worker_truck_consumption", 7) # l/100 km

        diesel_price = self.get_supply_param(
            "diesel_price", 0.65) # $/l (USA)

        unlock_fee = self.get_supply_param(
            "unlock_fee", 1) # $

        rent_fee = self.get_supply_param(
            "rent_fee", 0.15) # $/min

        avg_swap_distance = self.get_supply_param(
            "avg_swap_distance", 0.5) # km

        avg_trip_duration = self.get_supply_param(
            "avg_trip_duration", 10) # min

        swap_vehicle_cost_per_km = swap_vehicle_consumption * diesel_price

        # Lump sum costs for swap scheduling
        unitary_swap_cost = avg_swap_distance * swap_vehicle_cost_per_km
        unitary_scooter_revenue = unlock_fee + rent_fee*avg_trip_duration

        satisfied_charged_zones_idxs = []

        empty_queue = False

        try:
            # Get the top-priority zone
            _, zone_queue_idx = zones_priority_queue.get_nowait()
        except queue.Empty:
            return

        for i in range(n_free_workers):
            if empty_queue:
                break

            residual_capacity = battery_swap_capacity
            scheduled_swap = {}

            tot_swap_cost = 0
            tot_pote_revenues = 0

            n_battery_swaps = 0

            positive_income = False

            while residual_capacity > 0 and not empty_queue:
                if swap_profitability_check \
                        and n_battery_swaps > 0:
                    potential_lump_income = tot_pote_revenues - tot_swap_cost

                    # Terminate the scheduling stage for the i-th worker,
                    # if the updated scheduling would revert
                    # the positivity of the lump income
                    if potential_lump_income > 0:
                        positive_income = True
                    else:
                        if positive_income:
                            break

                try:
                    # Pick the next zone in the queue
                    zone_id    = starting_zones_ids[zone_queue_idx]
                    n_vehicles = n_dead_vehicles[zone_queue_idx]

                    # If the charging deficit is bigger than the residual worker capacity,
                    # try to fill it up as much as it possibly can...
                    if n_vehicles > residual_capacity:
                        n_battery_swaps = residual_capacity
                        n_dead_vehicles[zone_queue_idx] -= n_battery_swaps

                        # Clear satisfied zones
                        for idx in sorted(satisfied_charged_zones_idxs, reverse=True):
                            del starting_zones_ids[idx]
                            del n_dead_vehicles[idx]

                        satisfied_charged_zones_idxs.clear()

                        # Update zones charging deficit
                        starting_zones_ids, n_dead_vehicles = self.__update_zones_deficit(
                                                      starting_zones_ids, n_dead_vehicles)

                        # Update zones priority
                        zones_priority_queue = self.__compute_zones_priority(min_zone_id,
                                                                             starting_zones_ids)
                    else: # ...otherwise cover the whole charging deficit
                        n_battery_swaps = n_vehicles
                        n_dead_vehicles[zone_queue_idx] -= n_battery_swaps

                        satisfied_charged_zones_idxs.append(zone_queue_idx)

                    _, zone_queue_idx = zones_priority_queue.get_nowait()
                except queue.Empty:
                    empty_queue = True
                finally:
                    if zone_id not in scheduled_swap:
                        # TODO: Keep zone-only cost modelling?
                        scheduled_swap[zone_id] = n_battery_swaps
                        tot_swap_cost += unitary_swap_cost
                    else:
                        scheduled_swap[zone_id] += n_battery_swaps

                    residual_capacity -= n_battery_swaps
                    tot_pote_revenues += n_battery_swaps \
                                         * unitary_scooter_revenue

            # Null schedules filtering
            scheduled_swap = {
                k: v for k, v
                     in scheduled_swap.items()
                     if v > 0
            }

            if len(scheduled_swap) == 0:
                continue

            # Revert the i-th worker to idle, if its scheduled swap
            # would not generate a positive income
            if swap_profitability_check:
                if positive_income:
                    self.scheduled_swaps.append(scheduled_swap)
                    continue

            self.scheduled_swaps.append(scheduled_swap)

        self.__schedule_swaps(n_free_workers)

    def __schedule_swaps(self, n):
        """
        Schedule battery swap operations.

        Parameters
        ----------
        n : int
            Maximum number of scheduled swaps.
        """
        free_workers = [w for w in self.workers.items
                        if not w.busy]
        n_free_workers = len(free_workers)

        if n_free_workers == 0:
            return

        # Update n accordingly
        n = min(n, len(self.scheduled_swaps), n_free_workers)

        # Extract the workers dists by zone
        dists_by_zone = self.__compute_dists_by_zone(free_workers)

        # Assign the closest-to-the-1st-zone free worker
        # to each scheduled swap
        for swap in self.scheduled_swaps[:n]:
            zones_ids = list(swap.keys())
            first_zone_id = zones_ids[0]

            # Extract the nearest free worker
            worker_dists = dists_by_zone[first_zone_id]

            nearest_w = None
            nearest_w_dist = float('inf')

            for w in worker_dists.keys():
                if worker_dists[w] < nearest_w_dist \
                        and not w.busy:
                    nearest_w = w
                    nearest_w_dist = worker_dists[w]

            # Compute the shortest path...
            charging_path = self.__compute_shortest_path(first_zone_id,
                                                         zones_ids[1:])

            # ...+ the worker position
            charging_path = [nearest_w.current_position] + charging_path

            # Trigger multi-step battery swap
            self.env.process(self.charge_scooters_multi_zones(
                swap, charging_path, nearest_w))

    def __compute_shortest_path(self, starting_zone_id, zones_ids):
        """
        Compute the shortest path within a sequence of geo-localized zones by solving the
        Travelling Salesperson Problem (TSP) optimization problem with a genetic algorithm.
        """
        zones = [starting_zone_id] + zones_ids
        coords_list = []

        # Extract zones coordinates
        grid_h = self.sim_input.grid_matrix.shape[0]

        for zone in zones:
            c = int(np.floor(zone / grid_h))
            r = int(zone - c*grid_h)

            coords_list.append((c, r))

        # Solve the TSP optimization problem
        tsp_problem = TSPOpt(length=len(coords_list),
                             coords=coords_list,
                             maximize=False)

        best_path, _ = genetic_alg(tsp_problem, max_iters=10, random_state=2)
        best_path = deque(best_path)

        # Rotate the path to its best form until
        # the starting point P is not 0
        P = best_path[0]

        while P != 0:
            best_path.rotate(1)
            P = best_path[0]

        return [zones[i] for i in list(best_path)]

    def __compute_dists_by_zone(self, workers: list):
        """
        Compute the spatial distances to the starting zones of all scheduled swaps for a set of workers.
        """
        dists_by_zone = {}

        for (w, swap) in it.product(workers,
                                    self.scheduled_swaps):
            first_zone_id = list(swap.keys())[0]

            if first_zone_id not in dists_by_zone:
                dists_by_zone[first_zone_id] = {}

            dists_by_zone[first_zone_id][w] = self.get_distance(
                w.current_position,
                first_zone_id,
                False
            )

        return dists_by_zone

    def __update_zones_deficit(self, zones_ids, zones_deficit):
        """
        Update the charging deficit for each zone.
        """
        # Sort zones by charging deficit
        sorted_zones_deficit = sorted(zip(zones_ids, zones_deficit),
                                      key=lambda x: x[1])

        # Revert to descending order
        sorted_zones_deficit = sorted_zones_deficit[::-1]

        # Unzip zone Ids from charging deficits
        sorted_zones_deficit = list(zip(*sorted_zones_deficit))

        new_zones_ids     = list(sorted_zones_deficit[0])
        new_zones_deficit = list(sorted_zones_deficit[1])

        return new_zones_ids, new_zones_deficit

    def __compute_zones_priority(self, ref_zone_id, zones_ids):
        """
        Compute the zones priority as a weighted sum of their charging deficit and their spatial distance
        from the zone which is currently scoring the lowest charging deficit.

        The lower the score, the higher the priority.

        Parameters
        ----------
        ref_zone_id : int
            Zone Id with the lowest deficit.

        zones_ids : list
            Zone Ids to compute the priority of.

        Returns
        -------
        zones_priority_queue : PriorityQueue
            Zones priority queue
        """
        n_zones_ids = len(zones_ids)
        zones_priority_queue = queue.PriorityQueue()

        # Compute the spatial distance of all the zones
        # from the reference zone
        zones_dists = [self.get_distance(ref_zone_id, i, False)
                       for i in zones_ids]

        # Extract the maximum distance
        # + 0.001 to avoid division by 0
        max_dist = max(zones_dists) + 0.001

        for i in range(n_zones_ids):
            dist = zones_dists[i]

            priority = int((i / n_zones_ids)*100
                         + (dist / max_dist)*100)

            zones_priority_queue.put(
                (priority, i))

        return zones_priority_queue

    def __pick_starting_zones(self, n=-1):
        """
        Pick the n starting zones with the highest # of dead vehicles in them.

        Parameters
        ----------
        n : int
            Maximum number of zones to pick. The default is -1.
            Use -1 to pick them all.

        Returns
        -------
        starting_zone_ids : list
            Proposed starting zones Ids ordered by charging deficit.

        n_dead_vehicles : list
            Proposed number of vehicles to charge in each zone.

        min_zone_id : int
            Zone Id with the the lowest deficit.
        """
        starting_zones_ids = []
        n_dead_vehicles = []

        deficit_by_zone = self.__compute_zones_deficit()
        deficit_by_zone = sorted(deficit_by_zone.items(),
                                 key=lambda k: k[1])  # Sort in ascending order

        # Extract the zone with lowest deficit
        min_zone_id = deficit_by_zone[0][0] \
            if deficit_by_zone else -1

        # Convert back to dict
        deficit_by_zone = dict(deficit_by_zone)

        if n == -1:
            n = len(deficit_by_zone)

        n = min(n, len(deficit_by_zone))

        for _ in range(n):
            zone, deficit = deficit_by_zone.popitem()

            starting_zones_ids.append(zone)
            n_dead_vehicles.append(deficit)

        return starting_zones_ids, n_dead_vehicles, min_zone_id

    def __compute_zones_deficit(self):
        """
        Compute the charging deficit for each zone as the # of dead vehicles it contains.
        """
        dead_vehicles_by_zone = it.groupby(self.get_dead_vehicles(),
                                           lambda v: v.zone)

        return {k : len(list(v)) for k, v in dead_vehicles_by_zone}
