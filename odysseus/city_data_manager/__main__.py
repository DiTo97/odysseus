"""
Pre-processing steps for Austin dataset.

Beware that the austin_scooter_trips.py was modified with respect to the original repository
to account for the changes to the column names that have been applied to the Austin dataset

Beware that the original Austin dataset is disaggregated to increase the variability by choosing a random
point within each census tracts shape with get_random_point_from_shape from geospatial_utils.py

Beware the observed period is from 08/2018 to 09/2019 because earlier and later data are unstable
due to the recency of the service and the COVID spread, respectively.

1. Download the Austin dataset by launching the get_austin_data.py script (01/08/2018 - 30/09/2019)
2. Place the dataset in the folder (create subfolders if necessary):
    odysseus/city_data_manager/data/Austin/raw/trips/city_of_austin/
3. Download the Texas' census tracts dataset from
    https://catalog.data.gov/dataset/tiger-line-shapefile-2019-state-texas-current-census-tract-state-based
4. Place the census tracts dataset in the folder (create subfolders if necessary):
    odysseus/city_data_manager/data/Austin/raw/us_census_bureau/
5. Run the pre-processing script:
    python -m odysseus.city_data_manager -c <cities> -y <years> -m <months>
"""

import argparse
import datetime

from odysseus.city_data_manager.city_geo_trips.nyc_citi_bike_geo_trips import NewYorkCityBikeGeoTrips
from odysseus.city_data_manager.city_geo_trips.big_data_db_geo_trips import BigDataDBGeoTrips
from odysseus.city_data_manager.city_geo_trips.louisville_geo_trips import LouisvilleGeoTrips
from odysseus.city_data_manager.city_geo_trips.minneapolis_geo_trips import MinneapolisGeoTrips
from odysseus.city_data_manager.city_geo_trips.austin_geo_trips import AustinGeoTrips
from odysseus.city_data_manager.city_geo_trips.norfolk_geo_trips import NorfolkGeoTrips
from odysseus.city_data_manager.city_geo_trips.kansas_city_geo_trips import KansasCityGeoTrips
from odysseus.city_data_manager.city_geo_trips.chicago_geo_trips import ChicagoGeoTrips
from odysseus.city_data_manager.city_geo_trips.calgary_geo_trips import CalgaryGeoTrips


parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--cities", nargs="+",
    help="specify cities"
)

parser.add_argument(
    "-y", "--years", nargs="+",
    help="specify years"
)

parser.add_argument(
    "-m", "--months", nargs="+",
    help="specify months"
)

parser.add_argument(
    "-d", "--data_source_ids", nargs="+",
    help="specify data source ids"
)


parser.set_defaults(
    cities=["Louisville"],
    data_source_ids=["city_open_data"],
    years=["2019"],
    months=[int(i) for i in range(1, 13)],
)

args = parser.parse_args()

for city in args.cities:
    for data_source_id in args.data_source_ids:
        for year in args.years:
            for month in args.months:
                print(datetime.datetime.now(), city, data_source_id, year, month)

                if data_source_id == "citi_bike":
                    geo_trips_ds = NewYorkCityBikeGeoTrips(year=int(year), month=int(month))

                elif data_source_id == "big_data_db":
                    geo_trips_ds = BigDataDBGeoTrips(city, data_source_id, year=int(year), month=int(month))

                elif data_source_id == "city_open_data":
                    if city == "Louisville":
                        geo_trips_ds = LouisvilleGeoTrips(year=int(year), month=int(month))
                    elif city == "Minneapolis":
                        geo_trips_ds = MinneapolisGeoTrips(year=int(year), month=int(month))
                    elif city == "Austin":
                        geo_trips_ds = AustinGeoTrips(year=int(year), month=int(month))
                    elif city == "Norfolk":
                        geo_trips_ds = NorfolkGeoTrips(year=int(year), month=int(month))
                    elif city == "Kansas City":
                        geo_trips_ds = KansasCityGeoTrips(year=int(year), month=int(month))
                    elif city == "Chicago":
                        geo_trips_ds = ChicagoGeoTrips(year=int(year), month=int(month))
                    elif city == "Calgary":
                        geo_trips_ds = CalgaryGeoTrips(year=int(year), month=int(month))

                geo_trips_ds.get_trips_od_gdfs()
                geo_trips_ds.save_points_data()
                geo_trips_ds.save_trips()

print(datetime.datetime.now(), "Done")
