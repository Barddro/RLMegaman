import argparse
import os
import time
import numpy as np
from cynes.windowed import WindowedNES
import sdl2

FRAME_TIME = 1.0 / 60.0
ENEMIES_START = 0x06C2
ENEMIES_END = ENEMIES_START + 0x1F

state_num = 1000

BOSS_HP = 0x06C1

def main():
    enemytable = []
    parser = argparse.ArgumentParser(description="Collect MegaMan start states.")
    parser.add_argument("--rom", default="rom.nes", help="Path to the NES ROM file.")
    parser.add_argument("--states-dir", default="states/flashman", help="Directory to save state files.")
    args = parser.parse_args()

    if not os.path.isfile(args.rom):
        raise FileNotFoundError(f"ROM not found: '{args.rom}'.")

    os.makedirs(args.states_dir, exist_ok=True)

    def get_state():
        global state_num
        print(f"getting state {state_num}")

        HP = 0x06C0
        nes[HP] = 28  # refill HP before saving

        state = nes.save()

        path = f"{args.states_dir}/{state_num}.state"
        with open(path, "wb") as f:
            f.write(state.tobytes())

        state_num += 1
        return state

    def load_state(path):
        with open(path, "rb") as f:
            data = f.read()

        state = np.frombuffer(data, dtype=np.uint8).copy()
        nes.load(state)


    nes = WindowedNES(args.rom)
    load_state("./states/woodman/3.state")

    space_prev = False

    while not nes.should_close:
        #print(f"Boss HP: {nes[BOSS_HP]}")
        frame_start = time.perf_counter()

        nes.step()

        for i in range(len(enemytable)):
            #print(nes[ENEMIES_START + i])
            if(nes[ENEMIES_START + i] < enemytable[i]):
                print("ENEMY TOOK DAMAGE!")


        keys = sdl2.SDL_GetKeyboardState(None)
        space_now = keys[sdl2.SDL_SCANCODE_SPACE]
        if space_now and not space_prev:
            get_state()

        space_prev = space_now

        elapsed = time.perf_counter() - frame_start
        sleep_for = FRAME_TIME - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        enemytable = []
        for i in range(ENEMIES_START, ENEMIES_END + 1):
            enemytable.append(nes[i])


if __name__ == "__main__":
    main()