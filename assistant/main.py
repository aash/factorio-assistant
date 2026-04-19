import cv2
import collections
import time
from common import exit_hotkey, timeout
from mapar import Snail

from overlay import overlay
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v,--version', help='show version')
    args = parser.parse_args()  # noqa: F841

    with overlay() as ov, \
            Snail() as snail, \
            exit_hotkey(ahk=snail.ahk) as cmd_get, \
            timeout(1000) as is_not_timeout:
        r = snail.window_rect
        # r = MapParser.get_factorio_client_rect(ahk, FACTORIO_WINDOW_NAME)
        # if r is None:
        #     pytest.fail('could not get client rectangle')
        
        with ov.scene('tst') as s:
            s.rect(*r.xywh(), pen_color=(0, 255, 0, 255), pen_width=2)
            # s.rect(*snail.non_ui_rect.moved(r.x0, r.y0).xywh(), pen_color=(0, 255, 0, 255), pen_width=1)
            for uir in snail.ui_brects:
                s.rect(*uir.moved(r.x0, r.y0).xywh(), pen_color=(0, 255, 0, 255), pen_width=1)

        # camera = dxcam.create(
        #     backend="dxgi", # default Desktop Duplication backend
        #     processor_backend="cv2", # default OpenCV processor
        #     output_color="BGR"
        # )


        # Single frame (returns numpy array, BGR)
        
        # frame = camera.grab(region=r.xywh(), copy=True)

        # Continuous capture loop
        # camera.start(target_fps=60, region=r.xywh())
        # frame = camera.get_latest_frame()
        # cv2.imwrite('fr.png', frame)

        t0 = time.monotonic()
        tfps = collections.deque([0] * 60, maxlen=60)
        UNITS_PER_SECOND = 1000

        while is_not_timeout():
            t0fps = time.perf_counter()
            # img = camera.get_latest_frame_view()
            img = snail.wait_next_frame()
            img1 = cv2.resize(img, None, fx=0.25, fy=0.25)

            f, b = cv2.imencode('.png', img1)
            h, w, _ = img1.shape

            with ov.scene('frame') as ss:
                ss.image(r.x0, r.y0, w, h, png_bytes=b.tobytes())
            dtfps = int((time.perf_counter() - t0fps) * UNITS_PER_SECOND)
            tfps.appendleft(dtfps)
            with ov.scene('hud') as hud:
                t = time.monotonic() - t0
                fps = len(tfps) * UNITS_PER_SECOND / sum(tfps)
                ft = f'{t:6.3f}, FPS = {fps:06.1f}'
                hud.text(40, 80, ft, (0, 255, 0, 255), "JetBrainsMono NFM", 10)
            if cmd_get() == 'exit':
                break
            # time.sleep(0.05)
        # ahk.stop_hotkeys()
