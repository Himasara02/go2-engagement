"""Go2 AIR/PRO link over WebRTC (go2_webrtc_connect, legion1581).

One background asyncio thread owns the connection: it pushes the newest
camera frame into `latest_frame` and sends Move/StopMove commands queued by
the main (perception) thread. If the library or the robot is unreachable,
`Go2Link.ok` stays False and the caller falls back to a local camera.
"""
from __future__ import annotations

import asyncio
import threading
import time


class Go2Link:
    def __init__(self, ip: str | None, ap_mode: bool = False, video=True, control=True):
        self.ip, self.ap_mode = ip, ap_mode
        self.want_video, self.want_control = video, control
        self.latest_frame = None
        self.frame_t = 0.0
        self.ok = False
        self.err = None
        self._loop = asyncio.new_event_loop()
        self._conn = None
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        t0 = time.time()
        while not self.ok and self.err is None and time.time() - t0 < 25:
            time.sleep(0.2)

    # ---------------------------------------------------------------- thread
    def _run(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
            self.ok = True
            self._loop.run_forever()
        except Exception as e:     # noqa: BLE001
            self.err = repr(e)

    async def _connect(self):
        from go2_webrtc_driver.webrtc_driver import (Go2WebRTCConnection,
                                                     WebRTCConnectionMethod)
        if self.ap_mode:
            self._conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
        else:
            self._conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
        await self._conn.connect()
        if self.want_video:
            self._conn.video.switchVideoChannel(True)

            async def on_track(track):
                while True:
                    fr = await track.recv()
                    self.latest_frame = fr.to_ndarray(format="bgr24")
                    self.frame_t = time.time()
            self._conn.video.add_track_callback(on_track)

    # --------------------------------------------------------------- control
    async def _sport(self, api_name: str, param=None):
        from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
        req = {"api_id": SPORT_CMD[api_name]}
        if param is not None:
            req["parameter"] = param
        await self._conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], req)

    def _submit(self, coro):
        if self.ok and self.want_control:
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def move(self, vx: float, wz: float):
        self._submit(self._sport("Move", {"x": float(vx), "y": 0.0, "z": float(wz)}))

    def stop(self):
        self._submit(self._sport("StopMove"))

    def stand(self):
        self._submit(self._sport("BalanceStand"))
