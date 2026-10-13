"""Wall-clock admission control; in-flight searches also check expired()."""
import math
import time


class RunBudget:
    def __init__(self, started, seconds, minimum_reserve, reserve_factor, clock=time.monotonic):
        if not all(math.isfinite(x) for x in (started, seconds, minimum_reserve, reserve_factor)):
            raise ValueError('Budget values must be finite')
        if seconds < 0 or minimum_reserve < 0 or reserve_factor < 1:
            raise ValueError('Invalid run budget')
        self.deadline = float('inf') if seconds == 0 else started + seconds
        self.minimum_reserve = minimum_reserve
        self.reserve_factor = reserve_factor
        self.clock = clock
        self.history = []

    def expired(self):
        return self.clock() >= self.deadline

    def can_start(self):
        reserve = max(self.minimum_reserve, self.reserve_factor * max(self.history, default=0.))
        return self.deadline - self.clock() > reserve

    def observe(self, seconds):
        self.history.append(max(0., seconds))
        self.history = self.history[-8:]
