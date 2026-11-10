#!/usr/bin/env python3
"""
build_features.py
=================

Raw-data pipeline for the KAIST ball-bearing run-to-failure dataset
(Bronze -> Silver -> Gold, following the Medallion architecture of Lecture 8).

Dataset
-------
W. Jung, S.-H. Yun, Y.-H. Park, "Vibration, and temperature run-to-failure dataset of
ball bearing for prognostics", Data in Brief 54 (2024) 110403.
Mendeley Data, doi:10.17632/5hcdd3tdvb.6 (CC BY 4.0).

* 129 hourly CSV files ("Vibration_Bearing_RuntoFailure.zip"), hour 0 ... hour 128
* each file: 78.125 s sampled at 25.6 kHz  ->  2,000,000 rows x 4 columns
  vibration x (g), vibration y (g), bearing temperature (degC), ambient temperature (degC)
* NSK 6205 deep-groove ball bearing, ~1780 rpm, 2.94 kN axial + 5.88 kN radial load

What the pipeline produces (all small enough to live in a GitHub repo)
-----------------------------------------------------------------------
window_features.csv.gz     one row per 1-second window (~78 per hour, ~10,000 rows):
                           time-domain, frequency-domain and envelope (fault-frequency)
                           condition indicators for both axes + temperatures
file_quality_report.csv    one row per raw file: rows, missing values, duplicates,
                           z-score outliers, flat-lines, clipping, basic statistics
file_metadata.csv          one row per raw file: name, hour index, timestamp, shaft speed
psd_hourly.csv.gz          Welch power spectral density per hour and axis (25 Hz bins)
envelope_spectra_hourly.csv.gz  mean envelope spectrum per hour and axis (0-500 Hz, 1 Hz)
raw_excerpts.csv.gz        the first second of raw signal for selected hours
manifest.json              parameters, software versions, run time, row counts

Usage
-----
    python pipeline/build_features.py --download --out data/processed          # fetch from Mendeley
    python pipeline/build_features.py --zip Vibration_Bearing_RuntoFailure.zip  # local zip
    python pipeline/build_features.py --dir /path/to/csv_folder                 # extracted CSVs
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import json
import os
import platform
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

# --------------------------------------------------------------------------------------
# Constants (from the Data-in-Brief article and the Mendeley dataset description)
# --------------------------------------------------------------------------------------
FS = 25_600                      # sampling rate [Hz]
WIN_SEC = 1.0                    # aggregation window length [s]
WIN = int(FS * WIN_SEC)          # samples per window (25,600)
G = 9.80665                      # 1 g in m/s^2
START_TIME = "2022-06-20 17:00"  # start of the accelerated life test (dataset description)
NOMINAL_RPM = 1780.0             # rated test speed (1770-1780 rpm)

# NSK 6205 geometry: ball diameter d, pitch diameter D, number of balls N, contact angle 0
BALL_D, PITCH_D, N_BALLS = 7.90, 38.5, 9
_R = BALL_D / PITCH_D
FAULT_COEF = {                   # multiply by shaft frequency fr [Hz]
    "ftf": 0.5 * (1 - _R),                          # cage (fundamental train)
    "bpfo": N_BALLS / 2 * (1 - _R),                 # ball pass, outer race
    "bpfi": N_BALLS / 2 * (1 + _R),                 # ball pass, inner race
    "bsf": PITCH_D / (2 * BALL_D) * (1 - _R ** 2),  # ball spin
}

BANDS = {                        # vibration energy bands [Hz]
    "b1": (10, 1_000),           # shaft harmonics / structural
    "b2": (1_000, 3_000),
    "b3": (3_000, 6_000),        # typical housing resonances excited by defect impacts
    "b4": (6_000, 10_000),
    "b5": (10_000, 12_000),      # above the sensor's flat range (kept for completeness)
}
ENV_BAND = (1_000, 11_000)       # band-pass for envelope (demodulation) analysis [Hz]
ENV_FMAX = 500                   # envelope spectra stored up to this frequency [Hz]
PSD_NPERSEG = 8_192              # Welch segment length for hourly spectra (3.125 Hz)
PSD_BIN_HZ = 25                  # PSD re-binned to 25 Hz bins for storage
EXCERPT_HOURS = [0, 8, 16, 32, 48, 64, 80, 96, 104, 112, 116, 120, 122, 124, 125, 126, 127, 128]
EXCERPT_SEC = 1.0
CLIP_G = 49.5                    # PCB 352C34 range is +/-50 g

CANONICAL = ["vib_x", "vib_y", "temp_bearing", "temp_ambient"]

MENDELEY_ID, MENDELEY_VERSION = "5hcdd3tdvb", 6


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# 1. Download (Bronze layer)
# --------------------------------------------------------------------------------------
def _find_download_entries(obj, found=None):
    """Recursively collect (filename, url, size) triples from a Mendeley API response."""
    found = [] if found is None else found
    if isinstance(obj, dict):
        url = None
        if isinstance(obj.get("content_details"), dict):
            url = obj["content_details"].get("download_url")
            size = obj["content_details"].get("size")
        else:
            url = obj.get("download_url")
            size = obj.get("size")
        if url:
            name = obj.get("filename") or obj.get("name") or url.rsplit("/", 1)[-1]
            found.append((name, url, size))
        for v in obj.values():
            _find_download_entries(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _find_download_entries(v, found)
    return found


def download_dataset(workdir: Path) -> Path:
    """Download the dataset archive from Mendeley Data and return the local path."""
    import requests

    workdir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "ISE518-bearing-pipeline/1.0 (academic use)"
    candidates = []
    api_urls = [
        f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}/files?folder_id=root&version={MENDELEY_VERSION}",
        f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}?version={MENDELEY_VERSION}",
        f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}/files?version={MENDELEY_VERSION}",
    ]
    for u in api_urls:
        try:
            r = sess.get(u, timeout=60)
            log(f"API {u} -> HTTP {r.status_code}")
            if r.ok:
                js = r.json()
                log("API response (truncated): " + json.dumps(js)[:1500])
                entries = _find_download_entries(js)
                for name, url, size in entries:
                    log(f"  file entry: {name} | size={size} | {url}")
                candidates += [(n, u2, s) for n, u2, s in entries if str(n).lower().endswith(".zip")]
                if candidates:
                    break
        except Exception as exc:  # noqa: BLE001
            log(f"API {u} failed: {exc!r}")
    # "Download all" endpoints as a fallback
    candidates += [
        ("all.zip", f"https://data.mendeley.com/public-api/zip/{MENDELEY_ID}/download/{MENDELEY_VERSION}", None),
        ("all.zip", f"https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/{MENDELEY_ID}-{MENDELEY_VERSION}.zip", None),
    ]
    for name, url, size in candidates:
        dest = workdir / Path(str(name)).name
        try:
            log(f"Downloading {name} from {url}")
            with sess.get(url, stream=True, timeout=120, allow_redirects=True) as r:
                log(f"  HTTP {r.status_code}, content-type={r.headers.get('content-type')}, "
                    f"length={r.headers.get('content-length')}")
                if not r.ok:
                    continue
                done, t0, next_report = 0, time.time(), 512 * 2**20
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8 * 2**20):
                        fh.write(chunk)
                        done += len(chunk)
                        if done >= next_report:
                            log(f"  {done / 2**30:.2f} GiB ({done / 2**20 / (time.time() - t0):.0f} MiB/s)")
                            next_report += 512 * 2**20
            if zipfile.is_zipfile(dest):
                log(f"  saved {dest} ({dest.stat().st_size / 2**30:.2f} GiB)")
                return dest
            log("  downloaded file is not a zip archive, trying the next source")
        except Exception as exc:  # noqa: BLE001
            log(f"  download failed: {exc!r}")
    raise RuntimeError("Could not download the dataset from Mendeley Data. "
                       "Download Vibration_Bearing_RuntoFailure.zip manually and use --zip.")


# --------------------------------------------------------------------------------------
# 2. Discover and order the hourly CSV files
# --------------------------------------------------------------------------------------
_TS_RE = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _T-]*(\d{2})[-_.:h]?(\d{2})?")


def _parse_timestamp(name: str):
    m = _TS_RE.search(Path(name).stem)
    if not m:
        return None
    y, mo, d, h, mi = m.groups()
    try:
        return dt.datetime(int(y), int(mo), int(d), int(h), int(mi or 0))
    except ValueError:
        return None


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


class Source:
    """Uniform access to CSV files stored in a folder or in a (possibly nested) zip."""

    def __init__(self, path: Path, workdir: Path):
        self.path = Path(path)
        self.kind = "dir" if self.path.is_dir() else "zip"
        if self.kind == "zip":
            self.path = self._unnest(self.path, workdir)

    @staticmethod
    def _unnest(zpath: Path, workdir: Path) -> Path:
        """If the archive only wraps another zip (Mendeley 'download all'), extract it."""
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            csvs = [n for n in names if n.lower().endswith(".csv")]
            inner = [n for n in names if n.lower().endswith(".zip")]
            log(f"{zpath.name}: {len(csvs)} CSV and {len(inner)} zip members")
            if csvs or not inner:
                return zpath
            target = max(inner, key=lambda n: zf.getinfo(n).file_size)
            out = workdir / Path(target).name
            log(f"extracting inner archive {target} -> {out}")
            with zf.open(target) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 2**20)
        return Source._unnest(out, workdir)

    def list_csv(self):
        if self.kind == "dir":
            names = [str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.suffix.lower() == ".csv"]
        else:
            with zipfile.ZipFile(self.path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n]
        return names

    def open_bytes(self, name: str) -> bytes:
        if self.kind == "dir":
            return (self.path / name).read_bytes()
        with zipfile.ZipFile(self.path) as zf:
            return zf.read(name)


def order_files(names):
    """Sort files chronologically (timestamps in names if present, else natural order)."""
    stamps = {n: _parse_timestamp(n) for n in names}
    if all(stamps.values()):
        ordered = sorted(names, key=lambda n: stamps[n])
        t0 = stamps[ordered[0]]
        hours = [int(round((stamps[n] - t0).total_seconds() / 3600)) for n in ordered]
        how = "timestamp in file name"
    else:
        ordered = sorted(names, key=_natural_key)
        hours = list(range(len(ordered)))
        how = "natural sort of file names"
    return ordered, hours, how, stamps


# --------------------------------------------------------------------------------------
# 3. Robust CSV reader (Silver layer)
# --------------------------------------------------------------------------------------
def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_raw_csv(raw: bytes):
    """Parse one raw file into a DataFrame with canonical columns.

    Handles an optional header, optional metadata lines, comma/tab/semicolon/space
    delimiters and an optional leading time column. Returns (df, info).
    """
    head = raw[:65536].decode("utf-8", errors="replace").splitlines()
    delim = None
    for cand in [",", "\t", ";"]:
        if sum(line.count(cand) for line in head[:20]) >= 20:
            delim = cand
            break
    split = (lambda s: s.split(delim)) if delim else (lambda s: s.split())
    data_start = None
    for i, line in enumerate(head[:200]):
        toks = [t.strip().strip('"') for t in split(line) if t.strip() != ""]
        if len(toks) >= 4 and all(_is_float(t) for t in toks):
            data_start = i
            break
    if data_start is None:
        raise ValueError("no numeric data rows found in the first 200 lines")
    header = []
    if data_start > 0:
        header = [t.strip().strip('"') for t in split(head[data_start - 1])]
    n_cols = len([t for t in split(head[data_start]) if t.strip() != ""])
    df = pd.read_csv(io.BytesIO(raw), sep=delim if delim else r"\s+", header=None,
                     skiprows=data_start, engine="c" if delim else "python",
                     names=list(range(n_cols)), usecols=list(range(n_cols)),
                     on_bad_lines="skip")
    n_rows_read = len(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    info = {"header_line": " | ".join(header), "data_start_line": data_start,
            "n_columns_raw": n_cols, "delimiter": repr(delim) if delim else "whitespace"}

    # drop a leading time column (monotonic with constant step) if there are >4 columns
    cols = list(df.columns)
    if n_cols > 4:
        c0 = df[cols[0]].to_numpy()
        step = np.nanmedian(np.diff(c0[:10000]))
        if np.all(np.diff(c0[:10000]) > 0) and step > 0:
            info["dropped_time_column"] = True
            cols = cols[1:]
        cols = cols[:4]
    df = df[cols].copy()

    # map columns to canonical names (by header keywords if possible, else by position)
    names = [header[c].lower() if c < len(header) else "" for c in cols] if header else [""] * len(cols)
    mapping = {}
    axis_re = {a: re.compile(rf"(^|[^a-z]){a}([^a-z]|$)") for a in ("x", "y")}   # 'x' not inside 'axis'
    for c, nm in zip(cols, names):
        if "atmos" in nm or "ambient" in nm or "outdoor" in nm:
            mapping[c] = "temp_ambient"
        elif "bearing" in nm and "temp" in nm:
            mapping[c] = "temp_bearing"
        elif "vib" in nm and axis_re["x"].search(nm):
            mapping[c] = "vib_x"
        elif "vib" in nm and axis_re["y"].search(nm):
            mapping[c] = "vib_y"
    if sorted(mapping.values()) != sorted(CANONICAL):
        mapping = dict(zip(cols, CANONICAL))
        info["column_mapping"] = "positional"
    else:
        info["column_mapping"] = "header keywords"
    df = df.rename(columns=mapping)[CANONICAL]

    # sanity check on physical ranges: vibration ~0-mean, temperatures 0-150 degC
    means = df.mean()
    if not (abs(means["vib_x"]) < 2 and abs(means["vib_y"]) < 2 and 0 < means["temp_bearing"] < 200
            and 0 < means["temp_ambient"] < 200):
        info["range_warning"] = f"unexpected channel means: {means.round(3).to_dict()}"
    info["n_rows_read"] = n_rows_read
    return df, info


# --------------------------------------------------------------------------------------
# 4. Feature extraction (aggregation of 25.6 kHz samples into 1-s condition indicators)
# --------------------------------------------------------------------------------------
def estimate_shaft_frequency(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Shaft frequency from the 1x peak of a high-resolution spectrum (28-31 Hz search)."""
    nper = min(len(x), 2 ** 18)
    f, px = signal.welch(x, FS, nperseg=nper)
    _, py = signal.welch(y, FS, nperseg=nper)
    p = px + py
    band = np.flatnonzero((f >= 28.0) & (f <= 31.0))
    k = band[np.argmax(p[band])]
    # parabolic interpolation on the log-spectrum for sub-bin accuracy
    a, b, c = np.log(p[k - 1:k + 2] + 1e-30)
    delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    fr = float(f[k] + np.clip(delta, -0.5, 0.5) * (f[1] - f[0]))
    ref = np.median(p[(f >= 20) & (f <= 40)])
    prominence = float(p[k] / ref) if ref > 0 else np.nan
    if not np.isfinite(prominence) or prominence < 3:
        return NOMINAL_RPM / 60.0, prominence
    return fr, prominence


def _amp_at(freqs, amp, f0, tol=1.5):
    m = (freqs >= f0 - tol) & (freqs <= f0 + tol)
    return amp[:, m].max(axis=1) if m.any() else np.full(amp.shape[0], np.nan)


def window_features(sig: np.ndarray, env: np.ndarray, fr: float, prefix: str) -> dict:
    """Condition indicators for every 1-s window of one vibration channel.

    sig : raw acceleration [g], shape (n_win, WIN)
    env : envelope of the band-passed signal [g], shape (n_win, WIN)
    """
    out = {}
    mean = sig.mean(axis=1)
    x0 = sig - mean[:, None]                       # remove DC offset per window
    absx = np.abs(x0)
    rms = np.sqrt(np.mean(x0 ** 2, axis=1))
    peak = absx.max(axis=1)
    mean_abs = absx.mean(axis=1)
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_rms"] = rms
    out[f"{prefix}_peak"] = peak
    out[f"{prefix}_p2p"] = x0.max(axis=1) - x0.min(axis=1)
    out[f"{prefix}_crest"] = peak / rms
    out[f"{prefix}_kurtosis"] = stats.kurtosis(x0, axis=1, fisher=False)   # normal = 3
    out[f"{prefix}_skewness"] = stats.skew(x0, axis=1)
    out[f"{prefix}_shape"] = rms / mean_abs
    out[f"{prefix}_impulse"] = peak / mean_abs
    out[f"{prefix}_clearance"] = peak / np.mean(np.sqrt(absx), axis=1) ** 2

    # spectrum of each window (Hann window, one-sided PSD in g^2/Hz, 1 Hz resolution)
    w = np.hanning(WIN)
    spec = np.fft.rfft(x0 * w, axis=1)
    freqs = np.fft.rfftfreq(WIN, 1 / FS)
    psd = (np.abs(spec) ** 2) * 2.0 / (FS * np.sum(w ** 2))
    df_hz = freqs[1] - freqs[0]
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[f"{prefix}_{name}_rms"] = np.sqrt(psd[:, m].sum(axis=1) * df_hz)
    m = (freqs >= 10) & (freqs <= 12_000)
    out[f"{prefix}_spec_centroid"] = (psd[:, m] * freqs[m]).sum(axis=1) / psd[:, m].sum(axis=1)
    pn = psd[:, m] / psd[:, m].sum(axis=1, keepdims=True)
    out[f"{prefix}_spec_entropy"] = -(pn * np.log(pn + 1e-30)).sum(axis=1) / np.log(m.sum())
    # velocity RMS 10-1000 Hz in mm/s (ISO 10816 severity measure): V(f) = A(f) / (2*pi*f)
    mv = (freqs >= 10) & (freqs <= 1_000)
    vel_psd = psd[:, mv] * (G * 1000) ** 2 / (2 * np.pi * freqs[mv]) ** 2
    out[f"{prefix}_vel_rms"] = np.sqrt(vel_psd.sum(axis=1) * df_hz)
    # amplitude spectrum (g, peak) for discrete components
    amp = np.abs(spec) * 2.0 / np.sum(w)
    out[f"{prefix}_1x"] = _amp_at(freqs, amp, fr, tol=1.0)
    out[f"{prefix}_2x"] = _amp_at(freqs, amp, 2 * fr, tol=1.0)

    # envelope spectrum: fault-frequency amplitudes (sum of first 3 harmonics)
    e0 = env - env.mean(axis=1, keepdims=True)
    espec = np.abs(np.fft.rfft(e0 * w, axis=1)) * 2.0 / np.sum(w)
    noise = np.median(espec[:, (freqs >= 5) & (freqs <= 400)], axis=1)
    for key, coef in FAULT_COEF.items():
        f0 = coef * fr
        tot = sum(_amp_at(freqs, espec, h * f0, tol=1.5) for h in (1, 2, 3))
        out[f"{prefix}_env_{key}"] = tot
        out[f"{prefix}_env_{key}_snr"] = _amp_at(freqs, espec, f0, tol=1.5) / noise
    out[f"{prefix}_env_rms"] = np.sqrt(np.mean(e0 ** 2, axis=1))
    return out


def process_file(src_path: str, src_kind: str, name: str, hour: int, timestamp, workdir: str):
    """Read one raw file and return features, quality row, metadata row, spectra, excerpt."""
    t0 = time.time()
    src = Source.__new__(Source)
    src.path, src.kind = Path(src_path), src_kind
    raw = src.open_bytes(name)
    df, info = read_raw_csv(raw)
    del raw

    # ---------------- data-quality report on the full raw file ----------------
    q = {"hour": hour, "file": name, "n_rows": len(df), "duration_s": len(df) / FS}
    for c in CANONICAL:
        col = df[c]
        q[f"{c}_missing"] = int(col.isna().sum())
        q[f"{c}_mean"] = float(col.mean())
        q[f"{c}_std"] = float(col.std())
        q[f"{c}_min"] = float(col.min())
        q[f"{c}_max"] = float(col.max())
    q["duplicate_rows"] = int(df.duplicated().sum())
    for c in ["vib_x", "vib_y"]:
        v = df[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        z = (v - v.mean()) / v.std()
        q[f"{c}_z_gt3"] = int((np.abs(z) > 3).sum())
        q[f"{c}_z_gt5"] = int((np.abs(z) > 5).sum())
        q[f"{c}_clipped"] = int((np.abs(v) >= CLIP_G).sum())
        same = np.diff(v) == 0
        runs = np.diff(np.flatnonzero(np.concatenate(([True], ~same, [True])))) if same.any() else [1]
        q[f"{c}_max_flat_run"] = int(np.max(runs))
        q[f"{c}_skew"] = float(stats.skew(v))
        q[f"{c}_kurtosis"] = float(stats.kurtosis(v, fisher=False))
    q.update({k: v for k, v in info.items() if k != "n_rows_read"})

    # interpolate the (rare) missing samples so that signal processing can run
    arr = df.interpolate(limit_direction="both").to_numpy(dtype=np.float64)
    del df
    x, y, tb, ta = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    fr, prom = estimate_shaft_frequency(x, y)

    # envelope of the band-passed signal (whole file at once -> no window edge effects)
    sos = signal.butter(4, ENV_BAND, btype="bandpass", fs=FS, output="sos")
    envs = {}
    for k, s in (("x", x), ("y", y)):
        envs[k] = np.abs(signal.hilbert(signal.sosfiltfilt(sos, s)))

    n_win = len(x) // WIN
    sl = slice(0, n_win * WIN)
    feats = {"hour": np.full(n_win, hour), "window": np.arange(n_win),
             "t_in_file_s": np.arange(n_win) * WIN_SEC}
    for k, s in (("x", x), ("y", y)):
        feats.update(window_features(s[sl].reshape(n_win, WIN), envs[k][sl].reshape(n_win, WIN), fr, k))
    feats["temp_bearing"] = tb[sl].reshape(n_win, WIN).mean(axis=1)
    feats["temp_ambient"] = ta[sl].reshape(n_win, WIN).mean(axis=1)
    feats["shaft_hz"] = np.full(n_win, fr)
    fdf = pd.DataFrame(feats)

    # ---------------- hourly spectra ----------------
    f, px = signal.welch(x, FS, nperseg=PSD_NPERSEG)
    _, py = signal.welch(y, FS, nperseg=PSD_NPERSEG)
    edges = np.arange(0, FS / 2 + PSD_BIN_HZ, PSD_BIN_HZ)
    idx = np.digitize(f, edges) - 1
    nb = len(edges) - 1
    psd_rows = []
    for axis, p in (("x", px), ("y", py)):
        binned = np.bincount(idx, weights=p, minlength=nb + 1)[:nb] / np.maximum(np.bincount(idx, minlength=nb + 1)[:nb], 1)
        psd_rows.append([hour, axis] + binned.tolist())
    env_rows = []
    w = np.hanning(WIN)
    fe = np.fft.rfftfreq(WIN, 1 / FS)
    keep = fe <= ENV_FMAX
    for axis in ("x", "y"):
        e = envs[axis][sl].reshape(n_win, WIN)
        e = e - e.mean(axis=1, keepdims=True)
        es = (np.abs(np.fft.rfft(e * w, axis=1)) * 2.0 / np.sum(w)).mean(axis=0)
        env_rows.append([hour, axis] + es[keep].tolist())

    meta = {"hour": hour, "file": name,
            "timestamp": (timestamp.isoformat(sep=" ") if timestamp is not None else
                          (pd.Timestamp(START_TIME) + pd.Timedelta(hours=hour)).isoformat(sep=" ")),
            "shaft_hz": fr, "rpm": fr * 60, "speed_peak_prominence": prom,
            "n_windows": n_win, "process_seconds": round(time.time() - t0, 2)}

    excerpt = None
    if hour in EXCERPT_HOURS:
        n = int(EXCERPT_SEC * FS)
        excerpt = pd.DataFrame({"hour": hour, "t_s": np.arange(n) / FS, "vib_x": x[:n], "vib_y": y[:n],
                                "temp_bearing": tb[:n], "temp_ambient": ta[:n]})
    return fdf, q, meta, psd_rows, env_rows, excerpt


# --------------------------------------------------------------------------------------
# 5. Orchestration (Gold layer)
# --------------------------------------------------------------------------------------
def run(source: Source, out: Path, workdir: Path, workers: int = 1, limit: int | None = None):
    out.mkdir(parents=True, exist_ok=True)
    names = source.list_csv()
    if not names:
        raise RuntimeError("no CSV files found in the source")
    ordered, hours, how, stamps = order_files(names)
    log(f"{len(ordered)} CSV files, ordered by {how}; first={ordered[0]}, last={ordered[-1]}")
    if limit:
        ordered, hours = ordered[:limit], hours[:limit]
    jobs = [(str(source.path), source.kind, n, h, stamps.get(n), str(workdir)) for n, h in zip(ordered, hours)]

    results = []
    t0 = time.time()
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_file, *j): j for j in jobs}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                results.append(fut.result())
                log(f"processed {i}/{len(jobs)}: {futs[fut][2]} (hour {futs[fut][3]})")
    else:
        for i, j in enumerate(jobs, 1):
            results.append(process_file(*j))
            log(f"processed {i}/{len(jobs)}: {j[2]} (hour {j[3]}) in {results[-1][2]['process_seconds']} s")
    results.sort(key=lambda r: r[2]["hour"])

    feats = pd.concat([r[0] for r in results], ignore_index=True)
    quality = pd.DataFrame([r[1] for r in results])
    meta = pd.DataFrame([r[2] for r in results])
    edges = np.arange(0, FS / 2 + PSD_BIN_HZ, PSD_BIN_HZ)
    psd_cols = ["hour", "axis"] + [f"{int(lo)}" for lo in edges[:-1]]
    psd = pd.DataFrame([row for r in results for row in r[3]], columns=psd_cols)
    fe = np.fft.rfftfreq(WIN, 1 / FS)
    env_cols = ["hour", "axis"] + [f"{f:g}" for f in fe[fe <= ENV_FMAX]]
    env = pd.DataFrame([row for r in results for row in r[4]], columns=env_cols)
    excerpts = [r[5] for r in results if r[5] is not None]

    feats.to_csv(out / "window_features.csv.gz", index=False, float_format="%.6g")
    quality.to_csv(out / "file_quality_report.csv", index=False, float_format="%.6g")
    meta.to_csv(out / "file_metadata.csv", index=False, float_format="%.6g")
    psd.to_csv(out / "psd_hourly.csv.gz", index=False, float_format="%.5g")
    env.to_csv(out / "envelope_spectra_hourly.csv.gz", index=False, float_format="%.5g")
    if excerpts:
        pd.concat(excerpts, ignore_index=True).to_csv(out / "raw_excerpts.csv.gz", index=False, float_format="%.6g")

    manifest = {
        "dataset": "KAIST ball bearing run-to-failure (Mendeley Data doi:10.17632/5hcdd3tdvb.6, CC BY 4.0)",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(results), "file_ordering": how, "n_windows": int(len(feats)),
        "window_seconds": WIN_SEC, "fs_hz": FS, "envelope_band_hz": ENV_BAND, "bands_hz": BANDS,
        "fault_coefficients": FAULT_COEF, "excerpt_hours": EXCERPT_HOURS,
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"done: {len(feats)} windows from {len(results)} files in {manifest['runtime_seconds']} s -> {out}")
    return feats, quality, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", action="store_true", help="download the archive from Mendeley Data")
    g.add_argument("--zip", type=Path, help="path to Vibration_Bearing_RuntoFailure.zip")
    g.add_argument("--dir", type=Path, help="folder containing the extracted CSV files")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--workdir", type=Path, default=Path("raw_download"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (testing)")
    a = ap.parse_args(argv)
    a.workdir.mkdir(parents=True, exist_ok=True)
    path = download_dataset(a.workdir) if a.download else (a.zip or a.dir)
    run(Source(path, a.workdir), a.out, a.workdir, workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
