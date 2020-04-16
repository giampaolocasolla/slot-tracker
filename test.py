import csv
import random
import time
import datetime

timestamp = datetime.datetime.now()
cash = 800
total_2 = 1
total_3 = 0
total_4 = 0

fieldnames = ["timestamp", "cash", "total_2", "total_3", "total_4"]


with open("data.csv", "w") as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

while True:

    with open("data.csv", "a") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "timestamp": timestamp,
            "cash": cash,
            "total_2": total_2,
            "total_3": total_3,
            "total_4": total_4,
        }

        csv_writer.writerow(info)
        print(timestamp, cash, total_2, total_3, total_4)

        timestamp = datetime.datetime.now()
        cash += random.randint(-2, 1)
        total_2 = random.randint(1, 2)
        total_3 = random.randint(0, 4)
        total_4 = random.randint(0, 8)

    time.sleep(1)
