"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

# Imports
from collections import namedtuple
import numpy as np

from mlfinlab.data_structures.base_bars import BaseBars


class TimeBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, schema, threshold=50000, batch_size=20000000, to_csv=False):

        BaseBars.__init__(self, schema, metric='time', batch_size=batch_size)

        # Threshold at which to sample
        self.threshold = threshold
        # Named tuple to help with the cache
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'cum_volume', 'cum_dollar'])
        self.prev_timestamp = 0  # UTC format timestamp

    def _extract_bars(self, data):
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: Contains 3 columns - date_time, price, and volume.
        """
        cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = self._update_counters()
        data.iloc[:, self.schema.date_column] = data.iloc[:,
                                                          self.schema.date_column].astyp(int)  # convert to UTC timestamp

        # Iterate over rows
        list_bars = []
        for row in data.values:
            # Set variables
            date_time = row[self.schema.date_column]
            price = np.float(row[self.schema.price_column])
            volume = row[self.schema.volume_column]

            # Update high low prices
            high_price, low_price = self._update_high_low(
                high_price, low_price, price)

            # Calculations
            cum_ticks += 1
            dollar_value = price * volume
            cum_dollar_value = cum_dollar_value + dollar_value
            cum_volume += volume

            # Update cache
            self._update_cache(date_time, price, low_price,
                               high_price, cum_ticks, cum_volume, cum_dollar_value)

            time_diff = (date_time - self.prev_timestamp) / 1e9
            # If threshold reached then take a sample
            if time_diff >= self.threshold:  # pylint: disable=eval-used
                self._create_bars(date_time, price,
                                  high_price, low_price, list_bars)

                # Reset counters
                cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf
                self.cache = []
                self.prev_timestamp = date_time

                # Update cache after bar generation
                self._update_cache(
                    date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_dollar_value)

        return list_bars

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: Updated counters - cum_ticks, cum_dollar_value, cum_volume, high_price, low_price
        """
        # Check flag
        if self.flag and self.cache:
            last_entry = self.cache[-1]

            # Update variables based on cache
            cum_ticks = int(last_entry.cum_ticks)
            cum_dollar_value = np.float(last_entry.cum_dollar)
            cum_volume = last_entry.cum_volume
            low_price = np.float(last_entry.low)
            high_price = np.float(last_entry.high)
        else:
            # Reset counters
            cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf

        return cum_ticks, cum_dollar_value, cum_volume, high_price, low_price

    def _update_cache(self, date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_dollar_value):
        """
        Update the cache which is used to create a continuous flow of bars from one batch to the next.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param low_price: Lowest price in the period
        :param high_price: Highest price in the period
        :param cum_ticks: Cumulative number of ticks
        :param cum_volume: Cumulative volume
        :param cum_dollar_value: Cumulative dollar value
        """
        cache_data = self.cache_tuple(
            date_time, price, high_price, low_price, cum_ticks, cum_volume, cum_dollar_value)
        self.cache.append(cache_data)


def get_time_bars(schema, threshold=70000000, batch_size=20000000, verbose=True, to_csv=False):
    """
    Creates the dollar bars: date_time, open, high, low, close.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path: File path pointing to csv data.
    :param threshold: A cumulative value above this threshold triggers a sample to be taken.
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :return: Dataframe of dollar bars
    """

    bars = TimeBars(schema=schema, threshold=threshold, batch_size=batch_size)
    time_bars = bars.batch_run(verbose=verbose, to_csv=to_csv)
    return time_bars
