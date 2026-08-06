"""CanMV-specific adapters.

Imports are delayed so desktop tests do not require K230 firmware modules. The
camera initialization call is intentionally isolated here and will be verified
against the Yahboom example package and the connected board.
"""


class CanMVCamera:
    def __init__(self, width=640, height=480, channel=1):
        self.width = int(width)
        self.height = int(height)
        self.channel = int(channel)
        self.sensor = None

    def open(self):
        from media.sensor import Sensor, CAM_CHN_ID_1

        self.sensor = Sensor()
        self.sensor.reset()
        self.sensor.set_framesize(width=self.width, height=self.height, chn=self.channel)
        try:
            self.sensor.set_pixformat(Sensor.RGB565, chn=self.channel)
        except AttributeError:
            from media.sensor import PIXEL_FORMAT_RGB_565
            self.sensor.set_pixformat(PIXEL_FORMAT_RGB_565, chn=self.channel)
        self.sensor.run()

    def read(self):
        return self.sensor.snapshot(chn=self.channel)

    def close(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor = None


class CanMVUARTTransport:
    def __init__(self, uart_id=1, baudrate=115200):
        self.uart_id = int(uart_id)
        self.baudrate = int(baudrate)
        self.uart = None

    def open(self):
        from machine import UART

        self.uart = UART(self.uart_id, baudrate=self.baudrate)

    def write(self, message):
        self.uart.write(message)

    def close(self):
        if self.uart is not None:
            self.uart.deinit()
            self.uart = None


class CanMVClock:
    def millis(self):
        import time

        return time.ticks_ms()


class CanMVPipelineCamera:
    """Reuse Yahboom's verified camera/display/media pipeline."""

    def __init__(self, width=640, height=480, algorithm_width=320, algorithm_height=240, to_ide=True):
        self.width = int(width)
        self.height = int(height)
        self.algorithm_width = int(algorithm_width)
        self.algorithm_height = int(algorithm_height)
        self.to_ide = bool(to_ide)
        self.pipeline = None
        self.display_frozen = False
        self.display_api = None
        self.video_bind_info = None
        self.video_freeze_method = None

    def open(self):
        from libs.PipeLine import PipeLine

        self.pipeline = PipeLine(
            rgb888p_size=[self.algorithm_width, self.algorithm_height],
            display_size=[self.width, self.height],
            display_mode="lcd",
            osd_layer_num=4,
        )
        self.pipeline.create(ch1_frame_size=[self.width, self.height], to_ide=self.to_ide)
        from media.sensor import CAM_CHN_ID_0
        self.video_bind_info = self.pipeline.sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)

    def read(self):
        from media.sensor import CAM_CHN_ID_1

        return self.pipeline.sensor.snapshot(chn=CAM_CHN_ID_1)

    def freeze_display(self):
        """Unbind VIDEO1 while keeping the sensor/media pipeline running.

        PipeLine binds channel 0 directly to the LCD.  Merely stopping calls to
        snapshot() does not freeze that hardware video layer, so template
        authoring temporarily unbinds VIDEO1 while the OSD shows the captured
        frame and segmentation preview.  The sensor itself must remain running;
        repeated sensor.stop()/run() can block the next snapshot on K230.
        """
        if self.pipeline is not None and not self.display_frozen:
            if self.display_api is None:
                from media.display import Display
                self.display_api = Display
            unbind = getattr(self.display_api, "unbind_layer", None)
            disable = getattr(self.display_api, "disable_layer", None)
            if callable(unbind):
                result = unbind(self.display_api.LAYER_VIDEO1)
                if result is False:
                    raise RuntimeError("failed to unbind camera video layer")
                self.video_freeze_method = "unbind"
            elif callable(disable):
                disable(self.display_api.LAYER_VIDEO1)
                self.video_freeze_method = "disable"
            else:
                raise RuntimeError("firmware does not support video layer freeze")
            self.display_frozen = True

    def resume_display(self):
        if self.pipeline is not None and self.display_frozen:
            if self.display_api is None:
                from media.display import Display
                self.display_api = Display
            if self.video_bind_info is None:
                from media.sensor import CAM_CHN_ID_0
                self.video_bind_info = self.pipeline.sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
            self.display_api.bind_layer(
                **self.video_bind_info, layer=self.display_api.LAYER_VIDEO1
            )
            self.display_frozen = False
            self.video_freeze_method = None

    @property
    def overlay_image(self):
        return self.pipeline.osd_img

    def show_overlay(self):
        return self.pipeline.show_image()

    def close(self):
        if self.pipeline is not None:
            self.pipeline.destroy()
            self.pipeline = None
            self.display_frozen = False
            self.video_bind_info = None
            self.video_freeze_method = None


class ConsoleTransport:
    """Safe dry-run transport; writes results to the IDE/serial console only."""

    def open(self):
        return None

    def write(self, message):
        print(message, end="")

    def close(self):
        return None
