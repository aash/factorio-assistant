from mapar.snail import Snail
from mapar.snail import SnailWindowMode
from overlay_client import overlay_client
from common import exit_hotkey
from common import timeout
import time
import cv2

with overlay_client() as ovl_show_img, Snail(window_mode=SnailWindowMode.FULL_SCREEN) as s, \
        exit_hotkey(ahk=s.ahk) as cmd_get, \
        timeout(1000) as is_not_timeout:
    im = s.wait_next_frame()
    out = im.copy()

    t0 = time.time()
    idx = 0
    while is_not_timeout():
        im = s.wait_next_frame()
        if cmd_get() == 'exit':
            break

        if time.time() - t0 > 0.5:
            cv2.imwrite(f'fr_{idx:04d}.png', im)
            idx += 1
            t0 = time.time()
        time.sleep(0.010)




'''
import dxcam
import cv2

camera = dxcam.create(
    backend="dxgi", # default Desktop Duplication backend
    processor_backend="cv2", # default OpenCV processor
    output_color="BGR"
)


# Single frame (returns numpy array, BGR)
frame = camera.grab()

# Continuous capture loop
camera.start(target_fps=30)
frame = camera.get_latest_frame()
cv2.imwrite('fr.png', frame)

'''