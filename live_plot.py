
import glob
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

RESULT_DIR = os.path.join(os.getcwd(), "data", "result")
FILENAME = max(glob.glob(os.path.join(RESULT_DIR, "*")), key=os.path.getctime)

plt.style.use('ggplot')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]}) # sharex=True

def animate(i):
    data = pd.read_csv(FILENAME, parse_dates=["timestamp"]).dropna()
    x = data['timestamp']
    y = data['cash']

    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.cla()

    for ax in (ax1, ax2, ax3):
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for ax in (ax4, ax5):
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
    last_element = y.iloc[-1]
    rtp = np.round(100*last_element/y.iloc[0], 2)
    ax1.plot(x, y, label='All - RTP: '+ str(rtp))
    ax1.legend(loc='lower left')
    for ax, val in zip((ax2, ax3, ax4, ax5), [1000, 500, 100, 50]):
        if data.shape[0] >= val:
            rtp = np.round(100*last_element/y.iloc[-val], 2)
            ax.plot(x[-val:], y[-val:], label="Last " + str(val) + " - RTP: " + str(rtp))
            ax.legend(loc='lower left')

    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
