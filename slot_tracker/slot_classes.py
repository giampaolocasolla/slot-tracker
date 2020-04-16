import csv
import datetime
import logging
import os

import numpy as np
import pandas as pd

################################################################
# Path
################################################################

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(os.path.join(DATA_DIR, "log"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "result"), exist_ok=True)

################################################################
# Logger
################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(os.path.join(DATA_DIR, "log", "slot_tracker.log"))
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

################################################################
# Classes
################################################################


class Bet:
    def __init__(self, value, livelli, valori, total=0):
        self.value = value
        self.total = total
        self.livelli = livelli
        self.valori = valori
        logger.info("Calculate all the combinations")
        self.combinations = self.getCombinations(self.livelli, self.valori)
        logger.info("Select all the positions")
        self.positions = self.getPositions(self.combinations)
        self.possible_values = self.positions.index.tolist()

    def getCombinations(self, x_values, y_values):
        df = pd.DataFrame([x_values] * len(y_values), index=y_values, columns=x_values)
        df = df.multiply(10 * df.index, axis=0).round(1)
        return df

    def getPositions(self, whole_df):
        df = pd.DataFrame(
            index=np.unique(whole_df.values), columns=["VALORE", "LIVELLO"]
        )
        for i in df.index:
            y, x = np.where(whole_df.values.T == i)
            df.loc[i] = [x[0], y[0]]
        return df

    def changeBet(self, new_value, difference=False):

        if difference == False:
            if new_value in self.positions.index:
                change = self.positions.loc[new_value] - self.positions.loc[self.value]
            else:
                logger.error(f"Invalid new_value: {new_value}")
                raise Exception("Invalid new_value")

        elif difference == True:
            if new_value == int(new_value):
                current_index = np.where(self.positions.index == self.value)[0][0]
                if current_index + new_value < 0:
                    logger.error(f"new_value too low: {new_value}")
                    raise Exception("new_value too low")
                else:
                    change = (
                        self.positions.iloc[current_index + new_value]
                        - self.positions.loc[self.value]
                    )
            else:
                logger.error(f"Invalid new_value: {new_value}")
                raise Exception("Invalid new_value")

        else:
            logger.error(f"Invalid difference: {difference}")
            raise Exception("Invalid difference")

        return change.to_dict()


class Result:
    def __init__(self, cash, bet):
        self.timestamp = []
        self.cash = [cash]
        self.bet = [bet]
        self.gain = []
        self.gain_rel = []

    def timeNow(self):
        self.timestamp.append(datetime.datetime.now())

    def addGain(self, new_total):
        if new_total == np.nan:
            logger.warning("new_total is NaN so the gain will be NaN")
        result = np.round(new_total - self.cash[-1], 2)
        self.gain.append(result)
        self.gain_rel.append(np.round(result / self.bet[-1], 2))

    def getLastResult(self):
        result = {}
        try:
            for key, value in vars(self).items():
                result[key] = value[-1]
        except IndexError:
            logger.error("There are some attributes without values")
        else:
            logger.info(f"Last result: {result}")
            return result

    def saveResult(self, filename, result=None):
        if result is None:
            result = self.getLastResult()
        path = os.path.join(DATA_DIR, "result", filename)
        fieldnames = list(vars(self).keys())

        if not os.path.isfile(path):
            logger.info(f"There is no file {filename}")
            with open(path, "w") as f:
                writer = csv.DictWriter(f, delimiter=",", fieldnames=fieldnames)
                writer.writeheader()
            logger.info(
                f"Created file {filename} with the following fields: {fieldnames}"
            )

        with open(path, "a") as f:
            writer = csv.DictWriter(f, delimiter=",", fieldnames=fieldnames)
            writer.writerow(result)
            logger.info(f"Added last result to the file {filename}")
