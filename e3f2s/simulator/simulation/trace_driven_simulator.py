import datetime

import pandas as pd

from e3f2s.simulator.simulation.simulator import SharedMobilitySim
from e3f2s.utils.time_utils import update_req_time_info
from e3f2s.utils.vehicle_utils import *


class TraceDrivenSim (SharedMobilitySim):

    def init_data_structures(self):

        self.booking_request_arrival_rates = self.simInput.request_rates
        self.trip_kdes = self.simInput.trip_kdes
        self.od_distances = self.simInput.od_distances
        self.valid_zones = self.simInput.valid_zones

    def update_time_info(self):

        self.hours_spent += 1

        self.current_datetime = self.start + datetime.timedelta(seconds=self.env.now)
        self.current_hour = self.current_datetime.hour
        self.current_weekday = self.current_datetime.weekday()
        if self.current_weekday in [5, 6]:
            self.current_daytype = "weekend"
        else:
            self.current_daytype = "weekday"
    
    def mobility_requests_generator(self):

        self.init_data_structures()
        sim_timestamps = []

        for booking_request in self.simInput.booking_requests_list:

            self.vehicles_zones_history += [self.vehicles_zones]

            # TODO -> find faster alternative
            # self.n_vehicles_per_zones_history += [{
            #     zone: len(self.available_vehicles_dict[zone]) for zone in self.available_vehicles_dict
            # }]

            sim_timestamps += [self.current_datetime]

            self.update_time_info()
            booking_request = update_req_time_info(booking_request)
            booking_request["soc_delta"] = -get_soc_delta(booking_request["driving_distance"] / 1000)
            booking_request["soc_delta_kwh"] = soc_to_kwh(booking_request["soc_delta"])

            yield self.env.timeout(booking_request["ia_timeout"])
            self.process_booking_request(booking_request)

        self.n_vehicles_per_zones_history = pd.DataFrame(
            self.n_vehicles_per_zones_history,
            index=pd.DataFrame(self.sim_booking_requests)["start_time"]
        )