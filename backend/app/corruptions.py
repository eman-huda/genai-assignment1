"""Corruption definitions shared by Tasks 1, 2, 3 and the FastAPI backend.

Every function works on float32 RGB images in HWC layout with values in [0, 1].
NumPy and OpenCV are used (not torch) so the backend can reuse this file unchanged.
"""
import numpy as np
import cv2

CONDITIONS = ["clean", "salt_pepper", "gaussian_blur", "occlusion"]
COND2IDX = {c: i for i, c in enumerate(CONDITIONS)}
SEVERITIES = ["low", "medium", "high"]

# Training ranges (from the assignment brief)
SP_PROB_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_RECT_RANGE = (1, 3)
OCC_COVER_RANGE = (0.10, 0.35)

# Fixed test levels (from the assignment brief)
TEST_LEVELS = {
    "salt_pepper": [("low", {"p": 0.03}), ("medium", {"p": 0.08}), ("high", {"p": 0.15})],
    "gaussian_blur": [("low", {"kernel": 3, "sigma": 0.7}),
                      ("medium", {"kernel": 5, "sigma": 1.5}),
                      ("high", {"kernel": 7, "sigma": 2.5})],
    "occlusion": [("low", {"n_rects": 1, "coverage": 0.10}),
                  ("medium", {"n_rects": 2, "coverage": 0.20}),
                  ("high", {"n_rects": 3, "coverage": 0.35})],
}

# Severity bins for continuously sampled (validation) corruptions: thirds of each range
SEVERITY_EDGES = {
    "salt_pepper": (0.02 + 0.13 / 3, 0.02 + 2 * 0.13 / 3),        # on p
    "gaussian_blur": (0.5 + 2.0 / 3, 0.5 + 2 * 2.0 / 3),          # on sigma
    "occlusion": (0.10 + 0.25 / 3, 0.10 + 2 * 0.25 / 3),          # on coverage
}


def add_salt_pepper(img, p, seed):
    """Replace a fraction p of pixel locations with black or white (50/50)."""
    rng = np.random.default_rng(int(seed))
    h, w = img.shape[:2]
    out = img.copy()
    hit = rng.random((h, w)) < p
    salt = rng.random((h, w)) < 0.5
    out[hit & salt] = 1.0
    out[hit & ~salt] = 0.0
    return out


def add_gaussian_blur(img, kernel, sigma):
    k = int(kernel)
    return cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), sigmaX=float(sigma),
                            sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT_101)


def sample_rectangles(h, w, n_rects, coverage, rng, tol=0.015, bounds=None, max_tries=2000):
    """Place n_rects rectangles whose UNION covers approximately `coverage` of the image.

    Rejection sampling: area shares come from a Dirichlet draw, aspect ratios are
    log-uniform in [0.5, 2], positions are uniform. Returns ([[x, y, w, h], ...], achieved_coverage).
    """
    best, best_err, best_cov = None, np.inf, 0.0
    total = coverage * h * w
    for _ in range(max_tries):
        shares = rng.dirichlet(np.full(n_rects, 3.0)) if n_rects > 1 else np.array([1.0])
        mask = np.zeros((h, w), dtype=bool)
        rects = []
        for a in shares * total:
            ar = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            rh = int(np.clip(round(np.sqrt(a * ar)), 4, h))
            rw = int(np.clip(round(np.sqrt(a / ar)), 4, w))
            y0 = int(rng.integers(0, h - rh + 1))
            x0 = int(rng.integers(0, w - rw + 1))
            rects.append([x0, y0, rw, rh])
            mask[y0:y0 + rh, x0:x0 + rw] = True
        cov = float(mask.mean())
        err = abs(cov - coverage)
        in_bounds = bounds is None or (bounds[0] <= cov <= bounds[1])
        if in_bounds and err < best_err:
            best, best_err, best_cov = rects, err, cov
        if in_bounds and err <= tol:
            break
    return best, best_cov


def add_occlusion(img, rects):
    out = img.copy()
    for x, y, rw, rh in rects:
        out[y:y + rh, x:x + rw] = 0.0
    return out


def apply_corruption(img, params):
    """Apply a corruption described by a params dict (training sample or manifest entry)."""
    t = params["type"]
    if t == "clean":
        return img.copy()
    if t == "salt_pepper":
        return add_salt_pepper(img, params["p"], params["seed"])
    if t == "gaussian_blur":
        return add_gaussian_blur(img, params["kernel"], params["sigma"])
    if t == "occlusion":
        return add_occlusion(img, params["rects"])
    raise ValueError(f"Unknown corruption type: {t}")


def severity_bin(params):
    t = params["type"]
    if t == "clean":
        return "none"
    value = {"salt_pepper": params.get("p"), "gaussian_blur": params.get("sigma"),
             "occlusion": params.get("coverage")}[t]
    lo, hi = SEVERITY_EDGES[t]
    return "low" if value < lo else ("medium" if value < hi else "high")


def sample_params(condition, rng, h=128, w=128):
    """Sample one random corruption configuration from the TRAINING ranges."""
    seed = int(rng.integers(0, 2**31 - 1))
    if condition == "clean":
        params = {"type": "clean"}
    elif condition == "salt_pepper":
        params = {"type": "salt_pepper", "p": float(rng.uniform(*SP_PROB_RANGE))}
    elif condition == "gaussian_blur":
        params = {"type": "gaussian_blur", "kernel": int(rng.choice(BLUR_KERNELS)),
                  "sigma": float(rng.uniform(*BLUR_SIGMA_RANGE))}
    elif condition == "occlusion":
        n = int(rng.integers(OCC_RECT_RANGE[0], OCC_RECT_RANGE[1] + 1))
        target = float(rng.uniform(*OCC_COVER_RANGE))
        rects, cov = sample_rectangles(h, w, n, target, np.random.default_rng(seed),
                                       bounds=OCC_COVER_RANGE)
        params = {"type": "occlusion", "n_rects": n, "target_coverage": target,
                  "coverage": cov, "rects": rects}
    else:
        raise ValueError(condition)
    params["seed"] = seed
    params["severity"] = severity_bin(params)
    return params


def build_val_manifest(n_images, seed, h=128, w=128):
    """One deterministic, class-balanced corruption per validation image (training ranges)."""
    rng = np.random.default_rng(seed)
    conds = [CONDITIONS[i % 4] for i in range(n_images)]
    rng.shuffle(conds)
    entries = []
    for i, c in enumerate(conds):
        p = sample_params(c, rng, h, w)
        p["img_idx"] = i
        entries.append(p)
    return entries


def build_test_manifest(n_images, seed, h=128, w=128):
    """Every test image x (clean + 3 corruptions x 3 fixed severities) = 10 entries per image."""
    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n_images):
        entries.append({"img_idx": i, "type": "clean", "severity": "none",
                        "seed": int(rng.integers(0, 2**31 - 1))})
        for ctype, levels in TEST_LEVELS.items():
            for sev, cfg in levels:
                s = int(rng.integers(0, 2**31 - 1))
                e = {"img_idx": i, "type": ctype, "severity": sev, "seed": s, **cfg}
                if ctype == "occlusion":
                    rects, cov = sample_rectangles(h, w, cfg["n_rects"], cfg["coverage"],
                                                   np.random.default_rng(s))
                    e["target_coverage"] = cfg["coverage"]
                    e["coverage"] = cov
                    e["rects"] = rects
                entries.append(e)
    return entries