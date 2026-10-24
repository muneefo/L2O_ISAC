from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    nt: int = 16
    nr: int = 16
    users: int = 4
    targets: int = 2
    clutter: int = 2
    rf_chains: int = 8
    snr_db: float = 5.0
    noise_comm: float = 1.0
    noise_sense: float = 1.0
    rho: float = 0.8
    eps_h: float = 0.05
    eps_t: float = 0.05
    eps_r: float = 0.05
    target_variance_db: float = 2.0
    clutter_variance_db: float = 3.0
    angle_min_deg: float = -60.0
    angle_max_deg: float = 60.0
    min_separation_deg: float = 5.0

    def __post_init__(self):
        for name in ("nt", "nr", "users", "targets", "rf_chains"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.clutter < 0 or self.rf_chains > self.nt:
            raise ValueError("Require clutter >= 0 and rf_chains <= nt")
        if not 0 <= self.rho <= 1 or min(self.eps_h, self.eps_t, self.eps_r) < 0:
            raise ValueError("Invalid weight or uncertainty bound")
        if min(self.noise_comm, self.noise_sense) <= 0:
            raise ValueError("Noise variances must be positive")
        if self.angle_max_deg <= self.angle_min_deg:
            raise ValueError("Invalid angular range")
        if self.min_separation_deg < 0 or (self.paths - 1) * self.min_separation_deg >= self.angle_max_deg - self.angle_min_deg:
            raise ValueError("Infeasible angular separation")

    @property
    def streams(self):
        return self.users + self.targets

    @property
    def paths(self):
        return self.targets + self.clutter

    @property
    def power(self):
        return self.noise_comm * 10 ** (self.snr_db / 10)
