"""Orbbec RGB-D camera source (pyorbbecsdk, v1 or v2 API).

A background thread keeps only the newest (colour, depth) pair, so the
perception loop never falls behind the camera.

    cam = OrbbecSource()            # raises RuntimeError with a hint on failure
    ok, bgr, depth_mm, t = cam.read()
    cam.close()

`bgr` is uint8 HxWx3 (BGR). `depth_mm` is float32 HxW in millimetres,
aligned to the colour image (0 = no reading). Not mirrored.

Tested logic only against the pyorbbecsdk examples; if your model behaves
differently, run `python check_orbbec.py` and ask Claude Code to adapt this file.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


def _frame_to_bgr(frame):
    """Convert an Orbbec colour frame (any common format) to BGR."""
    from pyorbbecsdk import OBFormat
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.I420:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
    raise RuntimeError(f"unsupported colour format {fmt}")


class OrbbecSource:
    def __init__(self, width=1280, fps=30, timeout_s=10.0):
        try:
            import pyorbbecsdk as ob
        except ImportError as e:
            raise RuntimeError("pyorbbecsdk is not installed. See HANDOVER guide step 3.") from e
        self.ob = ob
        # on Windows, Pipeline() with no device crashes the process (0xC0000374)
        # instead of raising, so check for a camera first
        if ob.Context().query_devices().get_count() == 0:
            raise RuntimeError("no Orbbec camera found. Plug it into a USB-3 port "
                               "(re-plug it once) and run check_orbbec.py.")
        self.pipeline = ob.Pipeline()
        cfg = ob.Config()

        # colour: prefer RGB at the requested width, else the default profile
        cprofiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        cprof = None
        for w_, fmt, f_ in ((width, ob.OBFormat.RGB, fps), (0, ob.OBFormat.RGB, 0),
                            (width, ob.OBFormat.MJPG, fps)):
            try:
                cprof = cprofiles.get_video_stream_profile(w_, 0, fmt, f_)
                break
            except Exception:  # noqa: BLE001
                pass
        if cprof is None:
            cprof = cprofiles.get_default_video_stream_profile()
        cfg.enable_stream(cprof)
        dprofiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        cfg.enable_stream(dprofiles.get_default_video_stream_profile())

        # depth-to-colour alignment: v2 uses AlignFilter, v1 uses set_align_mode
        self.align = None
        if hasattr(ob, "AlignFilter"):
            self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            try:   # only hand over complete colour+depth pairs
                cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            except Exception:  # noqa: BLE001
                pass
        elif hasattr(ob, "OBAlignMode"):
            try:
                cfg.set_align_mode(ob.OBAlignMode.HW_MODE)
            except Exception:  # noqa: BLE001
                cfg.set_align_mode(ob.OBAlignMode.SW_MODE)
        try:
            self.pipeline.enable_frame_sync()
        except Exception:  # noqa: BLE001
            pass
        self.pipeline.start(cfg)
        self.profile = f"{cprof.get_width()}x{cprof.get_height()}@{cprof.get_fps()} {cprof.get_format()}"

        self._lock = threading.Lock()
        self._latest = None
        self._seen_t = 0.0
        self._run = True
        self.err = None
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        t0 = time.time()
        while self._latest is None and self.err is None and time.time() - t0 < timeout_s:
            time.sleep(0.05)
        if self._latest is None:
            self.close()
            raise RuntimeError(f"no frames from the Orbbec camera ({self.err}). "
                               "Check the USB-3 cable/port and run check_orbbec.py.")

    def _loop(self):
        while self._run:
            try:
                frames = self.pipeline.wait_for_frames(200)
                if frames is None:
                    continue
                if self.align is not None:
                    frames = self.align.process(frames)
                    if frames is None:
                        continue
                    frames = frames.as_frame_set()
                c, d = frames.get_color_frame(), frames.get_depth_frame()
                if c is None or d is None:
                    continue
                bgr = _frame_to_bgr(c)
                if bgr is None:
                    continue
                dw, dh = d.get_width(), d.get_height()
                depth = np.frombuffer(d.get_data(), dtype=np.uint16).reshape(dh, dw)
                depth = depth.astype(np.float32) * float(d.get_depth_scale())
                if depth.shape[:2] != bgr.shape[:2]:      # alignment not active
                    depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                with self._lock:
                    self._latest = (bgr, depth, time.time())
            except Exception as e:  # noqa: BLE001
                self.err = repr(e)
                time.sleep(0.05)

    def read(self):
        with self._lock:
            item = self._latest
        if item is None or item[2] == self._seen_t:
            return False, None, None, None
        self._seen_t = item[2]
        return True, item[0], item[1], item[2]

    def close(self):
        self._run = False
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
