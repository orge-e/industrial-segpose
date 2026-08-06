"""Desktop fakes used before a physical K230 is connected."""


class SequenceCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.opened = False

    def open(self):
        self.opened = True

    def read(self):
        if not self.opened or self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def close(self):
        self.opened = False


class SequenceDetector:
    def __init__(self, results):
        self.results = list(results)
        self.index = 0

    def detect(self, frame, timestamp_ms):
        if self.index >= len(self.results):
            return []
        result = self.results[self.index]
        self.index += 1
        return result


class MemoryTransport:
    def __init__(self):
        self.messages = []
        self.opened = False

    def open(self):
        self.opened = True

    def write(self, message):
        if not self.opened:
            raise RuntimeError("transport is closed")
        self.messages.append(message)

    def close(self):
        self.opened = False


class StepClock:
    def __init__(self, start=0, step=40):
        self.value = int(start) - int(step)
        self.step = int(step)

    def millis(self):
        self.value += self.step
        return self.value
