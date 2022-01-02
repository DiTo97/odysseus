import pandas as pd
import pathlib

Path = {
    'DATASET': pathlib.Path(__file__)     \
                   .absolute().parents[4] \
                   / 'demand_modelling'   \
                   / 'demand_models'      \
                   / 'Austin'             \
                   / 'bookings_train.csv',
}

if __name__ == '__main__':
    dataset = pd.read_csv(str(Path['DATASET']))

    n_uniq_vehicles = dataset['vehicle_id'].nunique()

    n_uniq_vehicles_by_year_month = dataset.groupby(['year', 'month']) \
                                           ['vehicle_id'].nunique().agg('mean')
