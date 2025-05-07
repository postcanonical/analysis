import numpy as np

class Calibration:
    def __init__(self, caption="Default"):
        self.caption = caption
        # Corresponds to C++: Amplitude, Voltage, Scale, DPhase
        self.amplitude_ref = 32768.0   # e.g. half‐scale
        self.voltage_ref   = 1.0
        self.scale_pct     = 1.0
        self.dphase_deg    = 0.0

        # Lists of (X, loss) points for amp & phase
        self.amp_points   = []   # [(amp0, loss0), (amp1, loss1), ...]
        self.phase_points = []   # [(phase0, loss0), (phase1, loss1), ...]

        # Calibration type flags (linear=0/quadratic=1)
        self.type_phase_internal = 0
        self.type_phase_external = 1

        # Phase‐loss lookup curve of size 361 (0..360°)
        self.phase_curve = np.zeros(361, dtype=float)
        self.phase_of_100_impacts = 0.0

        # Quadratic coefficients (filled by quad_interpolation)
        self.Q_A = self.Q_B = self.Q_C = 0.0

    def add_point(self, amp, phase, material_loss):
        """Add a combined amp/phase point (like C++ AddPoint(EPOINT))."""
        self.amp_points.append((amp, material_loss))
        self.phase_points.append((phase, material_loss))
        self._sort_points()

    def add_phase_point(self, phase, material_loss):
        """Add just a phase calibration point."""
        self.phase_points.append((phase, material_loss))
        self._sort_points()

    def _sort_points(self):
        self.amp_points   = sorted(self.amp_points, key=lambda p: p[0])
        self.phase_points = sorted(self.phase_points, key=lambda p: p[0])

    def get_phase_loss_lines(self, phase):
        """Linear interpolate between the two nearest phase_points."""
        pts = self.phase_points
        if len(pts) < 2:
            return 0.0
        # find first point ≥ phase
        for i, (ph, loss) in enumerate(pts):
            if ph >= phase:
                break
        if i == 0 or i >= len(pts):
            return 0.0
        x1, y1 = pts[i-1]
        x2, y2 = pts[i]
        return (y2 - y1)/(x2 - x1)*(phase - x1) + y1

    def get_amp_loss(self, amp):
        """Linear interpolation on amp_points (C++ GetAmpLoss)."""
        pts = self.amp_points
        if len(pts) < 2:
            return 0.0
        for i, (a, loss) in enumerate(pts):
            if a >= amp:
                break
        if i == 0 or i >= len(pts):
            return 0.0
        a1, l1 = pts[i-1]
        a2, l2 = pts[i]
        return (l2 - l1)/(a2 - a1)*(amp - a1) + l1

    def det3(self, m):
        """Compute 3×3 determinant (C++ Det3)."""
        return (
            m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
          - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
          + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0])
        )

    def quad_interpolation(self, Q1, Q2, Q3, value, condition):
        """
        Parabolic interpolation based on three phase_points,
        filling Q_A, Q_B, Q_C (C++ QuadInterpolation).
        """
        (x1, y1), (x2, y2), (x3, y3) = Q1, Q2, Q3
        # Build matrices for determinant solves...
        M = [[x1*x1, x1, 1],
             [x2*x2, x2, 1],
             [x3*x3, x3, 1]]
        det = self.det3(M)
        if det == 0:
            return 0.0
        # replace columns to get deta, detb, detc
        Ma = [[y1, x1, 1],
              [y2, x2, 1],
              [y3, x3, 1]]
        Mb = [[x1*x1, y1, 1],
              [x2*x2, y2, 1],
              [x3*x3, y3, 1]]
        Mc = [[x1*x1, x1, y1],
              [x2*x2, x2, y2],
              [x3*x3, x3, y3]]
        deta = self.det3(Ma)
        detb = self.det3(Mb)
        detc = self.det3(Mc)
        # store quadratic coeffs
        self.Q_A = deta/det
        self.Q_B = detb/det
        self.Q_C = detc/det

        # condition handling (DER_GZ / DER_LZ) omitted for brevity...
        return self.Q_A*value*value + self.Q_B*value + self.Q_C

    def recalc_phase_curve(self):
        """Recalculate full 0–360° loss curve (C++ RecalculatePhaseCurve)."""
        # zero it out
        self.phase_curve.fill(0.0)
        # internal defects
        if self.phase_of_100_impacts > 0:
            for i in range(int(self.phase_of_100_impacts)):
                self.phase_curve[i] = self.get_phase_loss_lines(float(i))
        # external defects via quadratic
        if self.type_phase_external == 1 and len(self.phase_points) >= 3:
            # find start index Q1 where phase_of_100_impacts falls
            pts = self.phase_points
            idx = next((j for j,(ph,_) in enumerate(pts) 
                        if ph >= self.phase_of_100_impacts), None)
            if idx is not None and idx+2 < len(pts):
                Q1, Q2, Q3 = pts[idx], pts[idx+1], pts[idx+2]
                # ensure reasonable quad fit
                err = self.quad_interpolation(Q1, Q2, Q3, Q1[0], condition=2)
                if abs(err - Q1[1]) < 0.1:
                    for i in range(int(self.phase_of_100_impacts), 361):
                        self.phase_curve[i] = (
                            self.Q_A*i*i + self.Q_B*i + self.Q_C
                        )

