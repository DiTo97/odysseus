class Zone(object):

    def __init__(self, env, zone_id, sim_start_time, vehicles):
        self.env = env
        self.zone_id = zone_id
        self.vehicles = vehicles

        self.current_status = dict()
        self.status_dict_list = list() # List of status over time (History)

        self.update_status(sim_start_time)

    def update_status(self, t):

        self.current_status = {
            "t": t,
            "vehicles_parked": len(self.vehicles),
        }
        self.status_dict_list.append(self.current_status)

    def add_vehicle(self, t):
        # self.vehicles references the corresponding list
        # from self.available_vehicles_dict in SharedMobilitySim
        #
        # The changes to that list will reflect also in
        # the self.vehicles variable; hence, there's no need to update the list here.
        self.update_status(t)

    def remove_vehicle(self, t):
        self.update_status(t)
