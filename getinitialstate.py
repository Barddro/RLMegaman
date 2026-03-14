from cynes import *
from cynes.windowed import WindowedNES
import pandas as pd
import numpy as np


def getState():
    save_state = nes.save()
    df = pd.DataFrame(save_state)
    df.to_csv('state.csv', index=False)


with WindowedNES("rom.nes") as nes:

    while not nes.should_close:
        mode = nes[0x075A]
        while mode != 2:
            mode = nes[0x075A]
            frame = nes.step()
        nes.controller = NES_INPUT_START | NES_INPUT_A
        frame = nes.step()
        if mode == 2:
            getState()
            break

