import pandas as pd
import numpy as np


class Bet:
    def __init__(self, value, livelli, valori):
        self.value = value
        self.livelli = livelli
        self.valori = valori
        self.combinations = self.get_combinations(self.livelli, self.valori)
        self.positions = self.get_positions(self.combinations)
        self.possible_values = self.positions.index.tolist()

    def get_combinations(self, x_values, y_values):
        df = pd.DataFrame([x_values] * len(y_values), index=y_values, columns=x_values)
        df = df.multiply(10 * df.index, axis=0).round(1)
        return df

    def get_positions(self, whole_df):
        df = pd.DataFrame(
            index=np.unique(whole_df.values), columns=["VALORE", "LIVELLO"]
        )
        for i in df.index:
            y, x = np.where(whole_df.values.T == i)
            df.loc[i] = [x[0], y[0]]
        return df

    def change_bet(self, new_value, difference=False):

        if difference == False:
            if new_value in self.positions.index:
                change = self.positions.loc[new_value] - self.positions.loc[self.value]
            else:
                raise Exception("Invalid new_value")

        elif difference == True:
            if new_value == int(new_value):
                current_index = np.where(self.positions.index == self.value)[0][0]
                if current_index + new_value < 0:
                    raise Exception("new_value too low")
                else:
                    change = (
                        self.positions.iloc[current_index + new_value]
                        - self.positions.loc[self.value]
                    )
            else:
                raise Exception("Invalid new_value")

        else:
            raise Exception("Invalid difference")

        return change.to_dict()
