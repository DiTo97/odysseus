import os

import pandas as pd
import numpy as np

from odysseus.city_data_manager.city_data_source.trips_data_source.trips_data_source import TripsDataSource


class AustinScooterTrips(TripsDataSource):

    def __init__(self):
        super().__init__("Austin", "city_of_austin", "e-scooter")

    def load_raw(self):

        raw_trips_data_path = os.path.join(
            self.raw_data_path,
            "Shared_Micromobility_Vehicle_Trips.csv"
        )

        self.trips_df = pd.read_csv(raw_trips_data_path,
                                    dtype={
                                        "council_district_start": np.str,
                                        "council_district_end": np.str,
                                        "census_geoid_start": np.str,
                                        "census_geoid_end": np.str
                                    },
                                    na_values={
                                        "council_district_start": "None",
                                        "council_district_end": "None",
                                        "census_geoid_start": ["OUT_OF_BOUNDS", "None"],
                                        "census_geoid_end": ["OUT_OF_BOUNDS", "None"]
                                    },
                                    parse_dates=[
                                        "start_time",
                                        "end_time",
                                        "modified_date"
                                    ])

        return self.trips_df

    def normalise(self, year, month):

        self.trips_df_norm = self.trips_df
        self.trips_df_norm = self.trips_df_norm.rename({
            "device_id": "vehicle_id",
            "trip_duration": "duration",
            "trip_distance": "driving_distance",
            "hour": "start_hour",
            "day_of_week": "start_weekday",
            "council_district_start": "start_council_district",
            "council_district_end": "end_council_district",
            "census_geoid_start": "start_census_tract",
            "census_geoid_end": "end_census_tract"
        }, axis=1)

        self.trips_df_norm = self.trips_df_norm[self.trips_df_norm.vehicle_type == "scooter"]

        self.trips_df_norm = self.trips_df_norm[[
            "trip_id",
            "vehicle_id",
            "start_time",
            "end_time",
            "year",
            "month",
            "start_hour",
            "duration",
            "start_census_tract",
            "end_census_tract",
            "driving_distance"
        ]]

        self.trips_df_norm = self.trips_df_norm[
            (self.trips_df_norm.year == year) & (self.trips_df_norm.month == month)
            ]

        self.trips_df_norm.dropna(inplace=True)

        self.trips_df_norm.start_census_tract = self.trips_df_norm.start_census_tract.astype(int)
        self.trips_df_norm.end_census_tract = self.trips_df_norm.end_census_tract.astype(int)

        self.trips_df_norm = super().normalise()

        self.save_norm()

        return self.trips_df_norm
