class DataSchema:
    def __init__(self, file_path, date_column, price_column, volume_column, tick_rule_column=None, threshold_column=None, tick_rule_dict=None):
        self.file_path = file_path
        self.date_column = date_column
        self.price_column = price_column
        self.volume_column = volume_column
        self.tick_rule_column = tick_rule_column
        self.threshold_column = threshold_column
        self.tick_rule_dict = tick_rule_dict

    def get_signed_tick(self, row):
        return self.tick_rule_dict[row[self.tick_rule_column]]
