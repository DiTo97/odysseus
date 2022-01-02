import simpy
import json
import datetime

from odysseus.supply_modelling.vehicle import Vehicle as Vehicle_definition


class Vehicle(Vehicle_definition):
    def __init__(self, env, plate, start_zone, start_soc,
                 vehicle_config, energymix_conf, sim_scenario_conf, sim_start_time):
        engine_type = sim_scenario_conf["engine_type"]
        model = sim_scenario_conf["vehicle_model_name"]

        if engine_type == "electric":
            energymix = energymix_conf
        else:
            energymix = {}

        # Get the vehicle config for "Electric"-"generic e scooter"
        super().__init__(vehicle_config[engine_type][model], energymix)

        self.env = env
        self.plate = plate
        self.zone = start_zone

        self.available = True
        self.soc = simpy.Container(env, init=start_soc, capacity=100) # soc: State of Charge

        self.current_status = {
            "time": sim_start_time,
            "status": "available",
            "soc": self.soc.level,
            "zone": self.zone
        }
        self.status_dict_list = [
            self.current_status
        ]

    def booking(self, booking_request):
        self.available = False

        self.current_status = {
            "time": booking_request["start_time"],
            "status": "booked",
            "soc": self.soc.level,
            "zone": self.zone
        }
        self.status_dict_list.append(self.current_status)

        yield self.env.timeout(booking_request["duration"])

        fuel_consumed = self.distance_to_consumption(booking_request["driving_distance"]/1000)
        percentage = self.consumption_to_percentage(fuel_consumed)

        self.soc.get(percentage)

        self.zone = booking_request["destination_id"]
        self.available = True

        self.current_status = {
            "time": booking_request["start_time"] + datetime.timedelta(seconds=booking_request['duration']),
            "status": "available",
            "soc": self.soc.level,
            "zone": booking_request["destination_id"]
        }
        self.status_dict_list.append(self.current_status)

    def charge(self, percentage):
        self.soc.put(percentage)

    def __str__(self):
        return json.dumps({
            'available': self.available,
            'capacity': self.capacity,
            'consumption': self.consumption,
            'engine_type': self.engine_type,
            'n_seats': self.n_seats,
            'plate': self.plate,
            'soc': int(self.soc.level)
        })
