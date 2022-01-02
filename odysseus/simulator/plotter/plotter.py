import os
import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt

Data = {
    'SIM_STATS': 'sim_stats.csv'
}


class Plotter:
    """
    Wrapper for experiments presented from Fig. 5 to Fig. 9 in the paper
    A. Ciociola, M. Cocca, et al. - E-Scooter Sharing: Leveraging Open Data for System Design
    """

    @staticmethod
    def plot_fleet_size(stats_dir: str, fig_name: str):
        """
        Figure 5: Satisfied demand % and average # of trips per e-scooter per month for a variable fleet size.

        Parameters
        ----------
        stats_dir : str
            Absolute path to the directory containing the results of multiple simulations.

        fig_name : str
            Filename of the figure to save.
        """

        cols_to_keep = [
            'n_bookings',
            'n_vehicles',
            'percentage_satisfied'
        ]

        df_stats = Plotter.get_concat_df(stats_dir)

        # Convert typings
        df_stats['n_vehicles'] = df_stats['n_vehicles'].astype(int)
        df_stats['n_bookings'] = df_stats['n_bookings'].astype(int)

        df_stats['percentage_satisfied'] = df_stats['percentage_satisfied'].astype(float)

        # Get sorted stacked DF
        df_stats = df_stats[cols_to_keep] \
            .sort_values('n_vehicles')

        # Compute avg trips per e-scooter
        df_stats['average_trips_per_e-scooter'] = (
                df_stats['n_bookings'].astype(float)
                / df_stats['n_vehicles'].astype(float)
        )

        X = df_stats[['n_vehicles']]

        Y1 = df_stats[['percentage_satisfied']]
        Y2 = df_stats[['average_trips_per_e-scooter']].astype(int)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(X, Y1, 'g', marker='o', label='Satisfied demand [%]')
        ax2.plot(X, Y2, 'b', marker='x', label='Average trips per e-scooter')

        ax1.set_xlabel('Number of e-scooters', fontsize=12)

        ax1.set_ylabel('Satisfied demand [%]', fontsize=12)
        ax2.set_ylabel('Average trips per e-scooter', fontsize=12)

        ax1.legend(loc='upper center',
                   bbox_to_anchor=(0.26, 1.09),
                   fancybox=True, shadow=True)
        ax2.legend(loc='upper center',
                   bbox_to_anchor=(0.74, 1.09),
                   fancybox=True, shadow=True)

        plt.subplots_adjust()
        plt.savefig(os.path.join(
            stats_dir,
            fig_name + '.png'))

    @staticmethod
    def plot_charging_thresh(stats_dir: str, fig_name: str):
        """
        Figure 6: Unsatisfied trips due to insufficient battery level %
        and # of needed battery swaps per 100 trips for a variable charging threshold $alpha$.

        Parameters
        ----------
        stats_dir : str
            Absolute path to the directory containing the results of multiple simulations.

        fig_name : str
            Filename of the figure to save.
        """
        cols_to_keep = [
            'alpha',
            'n_bookings',
            'n_charges',
            'percentage_not_enough_energy'
        ]

        df_stats = Plotter.get_concat_df(stats_dir)

        # Convert typings
        df_stats['alpha'] = df_stats['alpha'].astype(float)

        df_stats['n_bookings'] = df_stats['n_bookings'].astype(int)
        df_stats['n_charges'] = df_stats['n_charges'].astype(int)

        df_stats['percentage_not_enough_energy'] = df_stats['percentage_not_enough_energy'].astype(float)

        # Get sorted stacked DF
        df_stats = df_stats[cols_to_keep] \
            .sort_values('alpha')

        # Compute swaps per 100 trips
        df_stats['swaps_per_100_trips'] = (
                df_stats['n_charges'].astype(float) * 100
                / df_stats['n_bookings'].astype(float)
        )

        X = df_stats[['alpha']]

        Y1 = df_stats[['percentage_not_enough_energy']]
        Y2 = df_stats[['swaps_per_100_trips']].astype(int)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(X, Y1, 'g', marker='o', label='No energy trips [%]')
        ax2.plot(X, Y2, 'b', marker='x', label='Swaps per 100 trips')

        ax1.set_xlabel(r'$\alpha$', fontsize=12)

        ax1.set_ylabel('No energy trips [%]', fontsize=12)
        ax2.set_ylabel('Swaps per 100 trips', fontsize=12)

        ax1.legend(loc='upper center',
                   bbox_to_anchor=(0.26, 1.09),
                   fancybox=True, shadow=True)
        ax2.legend(loc='upper center',
                   bbox_to_anchor=(0.74, 1.09),
                   fancybox=True, shadow=True)

        plt.subplots_adjust()
        plt.savefig(os.path.join(
            stats_dir,
            fig_name + '.png'))

    @staticmethod
    def plot_charging_policy(stats_dir: str, fig_name: str):
        """
        Figure 7: Satisfied demand % for a variable # of workers and average reach time.

        Parameters
        ----------
        stats_dir : str
            Absolute path to the directory containing the results of multiple simulations.

        fig_name : str
            Filename of the figure to save.
        """
        cols_to_keep = [
            'avg_reach_time',
            'n_workers',
            'percentage_satisfied'
        ]

        df_stats = Plotter.get_concat_df(stats_dir)

        # Convert typings
        df_stats['avg_reach_time'] = df_stats['avg_reach_time'].astype(int)

        df_stats['n_workers'] = df_stats['n_workers'].astype(int)

        df_stats['percentage_satisfied'] = df_stats['percentage_satisfied'].astype(float)

        # Get unique avg reach times
        avg_reach_times = sorted(df_stats['avg_reach_time'].unique().tolist())

        # Get sorted stacked DF
        df_stats = df_stats[cols_to_keep]                \
            .groupby(['avg_reach_time'])

        colors = cm.rainbow(
            np.linspace(0, 1,
                        df_stats.ngroups))

        # Clear cache residuals
        plt.clf()

        for t, c in zip(avg_reach_times, colors):
            t_group = df_stats.get_group(t) \
                              .sort_values('n_workers')

            plt.plot(t_group['n_workers'],
                     t_group['percentage_satisfied'],
                     color=c, label=r'$t_{{reach}}$ = {} min'.format(t))

        plt.xlabel('Number of workers', fontsize=12)
        plt.ylabel('Satisfied demand [%]', fontsize=12)

        plt.grid()
        plt.legend()

        plt.savefig(os.path.join(
            stats_dir,
            fig_name + '.png'))

    @staticmethod
    def get_concat_df(stats_dir):
        """
        Load the concatenated Pandas DataFrame from all the subfolders.
        """
        df_stats = None

        for it in os.scandir(stats_dir):
            if it.is_dir():
                if df_stats is None:
                    df_stats = pd.read_csv(os.path.join(
                        it.path, Data['SIM_STATS']),
                        skiprows=1, header=None).T

                    df_stats = Plotter.firstrow2header(
                        df_stats)
                else:
                    df_single_stat = pd.read_csv(os.path.join(
                        it.path, Data['SIM_STATS']),
                        skiprows=1, header=None).T

                    df_single_stat = Plotter.firstrow2header(
                        df_single_stat)

                    df_stats = pd.concat([
                        df_stats,
                        df_single_stat],
                        ignore_index=True)

        return df_stats

    @staticmethod
    def firstrow2header(df):
        """
        Replace the header of a Pandas DataFrame with its 1st row.

        Returns
        -------
        df : pd.DataFrame
            Pandas DataFrame with updated header.
        """
        df.columns = df.iloc[0]
        df = df[1:]

        return df
