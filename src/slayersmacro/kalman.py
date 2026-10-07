"""1D constant-velocity Kalman filter for tracking the fishing marker.

Why this and not "read the pixel and press when it lines up":

Reading the pixel tells you where the marker WAS when the frame was
grabbed. By the time the key reaches the game, ~8.4ms has passed. On a fast
marker that is most of a pixel budget you care about, and reacting to a
measurement is always late.

A Kalman filter estimates position AND velocity, so we can ask where the
marker will be in t+8.4ms and act on that instead. It also smooths noisy
measurements, which matters because brightness thresholds flicker by a
pixel or two frame to frame.

State vector x = [position, velocity]. Constant-velocity motion, so:

    predict:  x = F x ,  P = F P F' + Q
    update:   y = z - H x ,  K = P H' (H P H' + R)^-1
              x = x + K y ,    P = (I - K H) P

H is [1 0] - we only measure position. The gain falls out as a 2x2
inverse, which is written out below rather than using np.linalg for speed.
"""

from __future__ import annotations

import numpy as np

# Latency we must compensate, milliseconds. Measured on this machine:
# ~4.2ms capture (vsync-locked at 240Hz) + ~4.2ms for the game to notice
# input on its next frame + ~0.3ms SendInput call. Rounded up.
SYSTEM_LATENCY_MS = 9.0


class MarkerTracker:
    """Tracks marker x-position in normalised 0..1 units with velocity.

    Feed it measurements with feed() as they arrive. Ask where it will be
    with predict_ahead() and decide using that rather than the last
    measurement.
    """

    def __init__(self,
                 dt: float = 1.0 / 240.0,
                 accel_sigma: float = 3.0,
                 meas_sigma: float = 5e-3) -> None:
        """
        dt: seconds between frames. 1/240 matches the vsync-locked capture.

        accel_sigma: std-dev of the marker's acceleration, normalised units
            per second squared. Too small and the filter trusts constant
            velocity so hard it lags through a direction change; too large
            and it becomes a running average.

        meas_sigma: std-dev of ONE detection, normalised units.

        Both are SIGMAS; the covariances below are their squares. Treating a
        variance as a sigma is a silent error - R too large makes the filter
        ignore measurements, which looks like a broken filter rather than a
        mistuned one.

        Defaults are not guesses. They came out of a 2-D sweep scored on
        lookahead error at the 9ms actuation horizon (the metric that
        decides whether a press lands), then checked for worst case rather
        than average:

            accel=3.0, meas=5e-3 -> nominal 0.00017
                                   worst over velocity 0.2..2.0  0.00021
                                   worst over noise 2e-4..2e-3    0.00054
                                   direction agreement after bounce 8/8

        Both sit interior to the sweep grid rather than on an edge, so they
        are a plateau and not a spike. For reference, meas=8e-4 scores worse
        on every axis despite being a smaller number: it under-trusts the
        real detector noise, which is what makes it lag.
        """
        self.dt = dt
        self.accel_sigma = accel_sigma
        self.meas_sigma = meas_sigma
        self.x = np.zeros(2)            # [position, velocity]
        self.P = np.eye(2) * 1.0        # wide = "I have no idea yet"
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        # Continuous white-noise-acceleration discretised for piecewise
        # constant acceleration over dt.
        sa2 = accel_sigma ** 2
        self.Q = sa2 * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                                 [dt ** 3 / 2.0, dt ** 2]])
        self.R = np.array([[meas_sigma ** 2]])
        self.initialised = False
        self.last_z = None

    def reset(self) -> None:
        self.__init__(self.dt, self.accel_sigma, self.meas_sigma)

    def feed(self, z: float) -> None:
        """One position measurement, normalised 0..1.

        Predict then update, exactly once per frame. Doing it twice was a
        real bug that made the filter trust the model less than intended.
        """
        if not self.initialised:
            # Seed from the first measurement with zero velocity. Starting
            # from zero matters: a bad first-frame guess otherwise throws
            # a large error that takes ~10 frames to wash out.
            self.x = np.array([z, 0.0])
            self.P = np.eye(2) * 0.01
            self.initialised = True
            self.last_z = z
            return
        self.predict()
        self.update(z)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float) -> None:
        # H @ x is shape (1,), so take the first element explicitly. A bare
        # float() on a 1-element array raises on numpy 2.x.
        y = z - float(np.ravel(self.H @ self.x)[0])
        # S is 1x1, so its inverse is 1/S. Written out to avoid a matrix
        # solve on the hot path.
        S = float(np.ravel(self.H @ self.P @ self.H.T)[0]) + float(self.R[0, 0])
        K = np.ravel(self.P @ self.H.T) / S      # shape (2,), not (2,1)
        self.x = np.ravel(self.x) + K * y        # ravel guards the shape
        # Joseph form: (I-KH) P (I-KH)' + K R K'. Numerically stabler than the
        # simplified (I-KH)P and it stays symmetric, which keeps P usable
        # for the variance estimate.
        H = np.ravel(self.H)
        IKH = np.eye(2) - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * float(self.R[0, 0])
        self.last_z = z

    @property
    def position(self) -> float:
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        """Normalised units per second."""
        return float(self.x[1])

    def predict_ahead(self, ms: float = SYSTEM_LATENCY_MS) -> float:
        """Where the marker will be after `ms`, at the current velocity.

        This is the whole point of the filter. Acting on `position` is
        always ~9ms stale by the time the game reacts.
        """
        steps = max(1, int(round((ms / 1000.0) / self.dt)))
        pred = self.x.copy()
        for _ in range(steps):
            pred = self.F @ pred
        return float(pred[0])

    def variance(self) -> float:
        return float(self.P[0, 0])

    def confident(self, threshold: float = 0.002) -> bool:
        """True once the estimate has settled.

        Without this we would act on the first noisy frame after a restart.
        """
        return self.initialised and self.variance() < threshold