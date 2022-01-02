import pandas as pd
from sodapy import Socrata

AustinAPI = {
    'APP_TOKEN': 's0PKCrJCcUhLpZGcA44ZCbB82',
    'DATASET'  : '7d8e-dm7r',
    'PASSWORD' : '7qK2bh7vnqVhiN@',
    'URI'      : 'data.austintexas.gov',
    'USERNAME' : 'fede97.minutoli@gmail.com'
}

if __name__ == '__main__':
    # # Unauthenticated client only works with public data sets. Note 'None'
    # # in place of application token, and no username or password:
    # client = Socrata(AustinAPI['URI'], None)

    client = Socrata(AustinAPI['URI'],
                     AustinAPI['APP_TOKEN'],
                     username=AustinAPI['USERNAME'],
                     password=AustinAPI['PASSWORD'])

    # Training data: 08/2018 - 08/2019
    # Test data: 09/2019
    wheres = {
        'vehicle_type': 'scooter',
        'where': "(start_time between \'2018-08-01T00:00:00\'"
                     " and \'2019-09-30T23:59:59\')"
                 " AND (end_time between \'2018-08-01T00:00:00\'"
                     " and \'2019-09-30T23:59:59\')",
        'limit': 50000000 # Generic big number
    }

    results = client.get(AustinAPI['DATASET'], **wheres)

    # Convert to pandas DataFrame
    results_df = pd.DataFrame.from_records(results)

    results_df.to_csv('./Shared_Micromobility_Vehicle_Trips.csv')
