from pynput import mouse
from pynput.mouse import Button, Controller


import ctypes


PROCESS_PER_MONITOR_DPI_AWARE = 2

ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

def on_move(x, y):
    print(
        f"moved.    {x:4d}, {y:4d}. "
    )

def on_click(x, y, button, pressed):
    print(
        f"clicked.  {x:4d}, {y:4d}. {button}, {pressed}"
    )
    
def on_scroll(x, y, dx, dy):
    print(
        f"scrolled {x:4d}, {y:4d}. {dx}, {dy}"
    )


# ...or, in a non-blocking fashion:
listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll)
listener.start()

while True :
    pass

print("terminating...")
listener.stop()
listener.join()
