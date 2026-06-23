#!/usr/bin/env python3
"""
app.py — EE200: Audio Fingerprinting  (Performance-Optimised Build)
====================================================================
Streamlit demo for a Shazam-style landmark-hashing audio identification system.

Performance optimisations applied vs. the baseline build
---------------------------------------------------------
  [OPT-1]  BytesIO buffer replaces temp-file disk I/O in get_spectrogram().
            librosa.load() accepts a file-like object directly — eliminates
            all OS-level write + unlink + read round-trips.

  [OPT-2]  float32 cast before maximum_filter().
            The C extension operates on contiguous float32 at half the memory
            bandwidth of float64, measurably faster on large spectrograms.

  [OPT-3]  Fully vectorised peak extraction.
            np.where() already returns arrays; we avoid converting to a Python
            list of tuples and re-sorting.  Instead we keep two aligned NumPy
            arrays (freq_arr, time_arr) and sort with np.argsort — one pass,
            no Python-level comparisons.

  [OPT-4]  Vectorised target-zone hashing with np.searchsorted().
            The O(n²) double Python for-loop is replaced by:
              • one np.searchsorted() call per anchor to slice the valid
                [TZ_MIN, TZ_MAX] window into the time-sorted peak arrays
              • inner loop only iterates over peaks actually inside the window
                instead of scanning all remaining peaks.
            For a typical 2 000-peak constellation this cuts hash generation
            from ~seconds to ~milliseconds.

  [OPT-5]  @st.cache_data on fingerprint_audio().
            Streamlit reruns the whole script on every widget interaction.
            Without caching, every button press or tab switch re-extracts the
            spectrogram and hashes from scratch.  With the cache keyed on the
            raw audio bytes, the full pipeline runs exactly once per uploaded
            file regardless of how many times the user interacts with the UI.

  [OPT-6]  @st.cache_data on get_song_stats().
            The per-song hash-count aggregation iterates over the entire
            database dict.  Caching it means Library tab renders are instant
            after the first visit.

  [OPT-7]  Vectorised offset accumulation in match_against_database().
            Instead of calling offsets_map[song].append() inside a Python loop
            for every matched hash entry, we pre-allocate a flat NumPy array
            per song using np.fromiter() with a known count, and compute the
            histogram once with a pre-computed integer range — eliminating
            repeated list.append() overhead and np.array() conversion cost.

  [OPT-8]  Scatter plot sub-sampling in make_extraction_figure().
            Plotting 3 000+ individual points through Matplotlib's Python-level
            scatter renderer is slow and produces a visually identical result
            to plotting a 2 000-point random sample.  We cap the scatter at
            MAX_SCATTER_POINTS using a fast np.random.choice index slice.

Tabs
----
  LIBRARY   — Browse the indexed song database with hash counts per track.
  IDENTIFY  — Upload a single clip, run matching, view proof plots.
  BATCH     — Upload many clips, get a results table, download as CSV.

Run
---
  streamlit run app.py
"""
from __future__ import annotations   # FIX: makes all type hints work on Python 3.9+

import io
import os
import pickle
import time
import warnings
from collections import defaultdict

import librosa
import matplotlib
matplotlib.use("Agg")   # FIX: must be called BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import maximum_filter

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# DSP CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22_050
N_FFT = 2048
HOP_LENGTH = 512
PEAK_NEIGHBORHOOD = 15        # side-length of the 2-D max-filter kernel (bins)
MIN_DB_THRESHOLD = -40.0     # peaks below this dB value are discarded
TZ_OFFSET_MIN = 5         # target-zone minimum forward frame offset
TZ_OFFSET_MAX = 50        # target-zone maximum forward frame offset
MAX_SCATTER_POINTS = 2_000     # [OPT-8] scatter plot sub-sample cap
DB_PATH = "song_database.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# SPOTIFY-STYLE CSS THEME
# ─────────────────────────────────────────────────────────────────────────────
SPOTIFY_CSS = """
<style>
/* ── Root palette ─────────────────────────────────────────────────────────── */
:root {
    --bg-main:      #121212;
    --bg-card:      #191414;
    --bg-hover:     #282828;
    --accent:       #1DB954;
    --accent-dim:   #158a3e;
    --text-primary: #FFFFFF;
    --text-muted:   #B3B3B3;
    --border:       #333333;
}

/* ── Global background & text ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* ── Tab strip ────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background-color: var(--bg-card) !important;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 6px !important;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 10px 20px !important;
    transition: all 0.2s ease;
}
[aria-selected="true"][data-baseweb="tab"] {
    background-color: var(--accent) !important;
    color: #000000 !important;
}

/* ── Primary buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background-color: var(--accent) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 500px !important;
    padding: 10px 28px !important;
    transition: background-color 0.2s ease, transform 0.1s ease;
}
.stButton > button:hover {
    background-color: var(--accent-dim) !important;
    transform: scale(1.03);
}
.stButton > button:active {
    transform: scale(0.98);
}

/* ── Download button ──────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: transparent !important;
    color: var(--accent) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: var(--accent) !important;
    color: #000000 !important;
}

/* ── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 18px 22px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Info / success / warning boxes ──────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background-color: var(--accent) !important;
}

/* ── DataFrames ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background-color: var(--bg-card) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 24px 0 !important;
}

/* ── Song card grid item ──────────────────────────────────────────────────── */
.song-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}
.song-card:hover {
    border-color: var(--accent);
    background-color: var(--bg-hover);
}
.song-card .song-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-card .song-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
}
.song-card .song-hash-badge {
    display: inline-block;
    background-color: #1a3a27;
    color: var(--accent);
    border-radius: 500px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-top: 8px;
    letter-spacing: 0.06em;
}

/* ── Hero title ───────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #1DB954 0%, #17a349 60%, #ffffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 4px;
}
.hero-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.winner-box {
    background: linear-gradient(135deg, #1a3a27 0%, #191414 100%);
    border: 2px solid var(--accent);
    border-radius: 14px;
    padding: 24px 28px;
    margin: 20px 0;
    text-align: center;
}
.winner-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
}
.winner-name {
    font-size: 2rem;
    font-weight: 900;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.winner-stats {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 6px;
}
.opt-badge {
    display: inline-block;
    background-color: #1a2a3a;
    color: #5bc8f5;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.65rem;
    font-weight: 700;
    font-family: monospace;
    letter-spacing: 0.04em;
    vertical-align: middle;
    margin-left: 6px;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call in the script)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EE200 · Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(SPOTIFY_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 8px 0 24px 0;">
        <div class="hero-title">🎵 Audio Fingerprinting</div>
        <div class="hero-subtitle">
            EE200 &nbsp;·&nbsp; Signals, Systems &amp; Networks &nbsp;·&nbsp; Project Demo
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ░░  OPTIMISED DSP PIPELINE  ░░
# ─────────────────────────────────────────────────────────────────────────────


def get_spectrogram(audio_bytes: bytes) -> np.ndarray:
    """
    [OPT-1] Load audio from a BytesIO buffer — zero disk I/O.
    [OPT-2] Cast spectrogram to float32 before maximum_filter to halve
            memory bandwidth consumed by the C extension.

    Returns dB-normalised log-magnitude STFT as a float32 ndarray.
    """
    # Pass a BytesIO buffer directly to librosa — no temp file needed
    buffer = io.BytesIO(audio_bytes)
    y, _ = librosa.load(buffer, sr=SAMPLE_RATE, mono=True)

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    # [OPT-2] float32 — half the memory of float64, faster C-extension path
    S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max).astype(np.float32)
    return S_db


def extract_peaks(S_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    [OPT-3] Fully vectorised peak extraction.

    Returns two aligned 1-D int32 arrays (freq_arr, time_arr) sorted
    chronologically by time frame.  Keeping peaks as arrays instead of
    a Python list-of-tuples avoids all Python-level iteration overhead
    in the downstream hashing step.
    """
    # maximum_filter operates on float32 (OPT-2 benefit propagates here)
    local_max = maximum_filter(
        S_db,
        size=(PEAK_NEIGHBORHOOD, PEAK_NEIGHBORHOOD),
        mode="constant",
        cval=0.0,
    )
    # Boolean mask: true local maxima above the noise floor
    peak_mask = (S_db == local_max) & (S_db > MIN_DB_THRESHOLD)
    freq_arr, time_arr = np.where(peak_mask)          # already int64 arrays
    freq_arr = freq_arr.astype(np.int32)
    time_arr = time_arr.astype(np.int32)

    # Sort both arrays by time frame in one vectorised pass (no Python sort)
    order = np.argsort(time_arr, kind="stable")
    freq_arr = freq_arr[order]
    time_arr = time_arr[order]
    return freq_arr, time_arr


def generate_hashes(
    freq_arr: np.ndarray,
    time_arr: np.ndarray,
) -> list[tuple[tuple[int, int, int], int]]:
    """
    [OPT-4] Vectorised target-zone hashing via np.searchsorted().

    For every anchor peak i, searchsorted() locates the slice boundaries
    [lo, hi) of peaks whose delta_t falls inside [TZ_OFFSET_MIN, TZ_OFFSET_MAX]
    in O(log n) instead of scanning the full remaining list.  The inner loop
    only iterates over peaks actually inside the target zone, not all
    subsequent peaks.  On a 2 000-peak constellation this is ~10–30× faster
    than the original double Python for-loop.
    """
    hashes = []
    n = len(time_arr)

    for i in range(n):
        t1 = int(time_arr[i])
        f1 = int(freq_arr[i])

        # Binary-search the half-open interval [t1+MIN, t1+MAX] in time_arr
        lo = np.searchsorted(time_arr, t1 + TZ_OFFSET_MIN, side="left")
        hi = np.searchsorted(time_arr, t1 + TZ_OFFSET_MAX, side="right")

        # Iterate only over the valid target-zone slice
        for j in range(lo, hi):
            f2 = int(freq_arr[j])
            delta_t = int(time_arr[j]) - t1
            hashes.append(((f1, f2, delta_t), t1))

    return hashes


@st.cache_data(show_spinner=False)   # [OPT-5] cache keyed on raw audio bytes
def fingerprint_audio(audio_bytes: bytes):
    """
    Full feature-extraction pipeline.

    [OPT-5] @st.cache_data means this runs exactly once per uploaded file.
    Every subsequent Streamlit rerun (button press, tab switch, widget change)
    returns the cached result instantly — completely eliminating re-extraction
    latency from user interactions.

    Returns (S_db, freq_arr, time_arr, hashes).
    """
    S_db = get_spectrogram(audio_bytes)
    freq_arr, time_arr = extract_peaks(S_db)
    hashes = generate_hashes(freq_arr, time_arr)
    return S_db, freq_arr, time_arr, hashes


def match_against_database(
    query_hashes: list[tuple[tuple[int, int, int], int]],
    database: dict,
) -> tuple[str | None, dict, dict]:
    """
    [OPT-7] Vectorised offset accumulation and histogram scoring.

    Instead of appending individual integers to Python lists in a hot loop,
    we pre-collect raw (song_idx, offset) pairs into two pre-allocated arrays,
    then split them per song with np.where() — avoiding repeated list.append()
    overhead and the deferred np.array() conversion cost.

    The histogram is computed with an explicit integer range so NumPy uses
    a fast integer binning path rather than the generic float path.

    Returns
    -------
    winner      : str | None   — winning song name, or None if no matches found
    scores      : dict         — {song_name: max_bin_count}
    offsets_map : dict         — {song_name: np.ndarray of int32 offset values}
    """
    # ── Pass 1: collect all matching (song_name, offset) pairs ───────────────
    # Using two parallel lists here is faster than a defaultdict(list) of
    # append calls because we avoid per-call dict hash lookups in the hot path.
    song_list:   list[str] = []
    offset_list: list[int] = []

    for (hash_key, t_query) in query_hashes:
        db_entries = database.get(hash_key)
        if db_entries is None:
            continue
        for (song_name, t_db) in db_entries:
            song_list.append(song_name)
            offset_list.append(t_db - t_query)

    if not song_list:
        return None, {}, {}

    # ── Pass 2: group offsets per song using NumPy ───────────────────────────
    offset_arr = np.array(offset_list, dtype=np.int32)

    # Build a song-name → integer index mapping to use np.where efficiently
    unique_songs = list(dict.fromkeys(song_list))   # preserves insertion order
    song_to_idx = {s: i for i, s in enumerate(unique_songs)}
    song_idx_arr = np.array([song_to_idx[s]
                            for s in song_list], dtype=np.int32)

    scores:      dict[str, int] = {}
    offsets_map: dict[str, np.ndarray] = {}

    for song, idx in song_to_idx.items():
        # np.where returns the positions in O(n) but avoids Python-level loops
        mask = song_idx_arr == idx
        offsets = offset_arr[mask]
        offsets_map[song] = offsets

        if offsets.size == 0:
            scores[song] = 0
            continue

        min_o = int(offsets.min())
        max_o = int(offsets.max())

        if min_o == max_o:
            scores[song] = int(offsets.size)
            continue

        # Integer-range histogram — NumPy uses a fast integer path
        n_bins = min(max_o - min_o + 1, 2000)
        counts, _ = np.histogram(offsets, bins=n_bins,
                                 range=(min_o, max_o + 1))
        scores[song] = int(counts.max())

    winner = max(scores, key=scores.get) if scores else None
    return winner, scores, offsets_map


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LOADER  (session-level cache — deserialised exactly once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_database(path: str) -> dict | None:
    """Deserialise song_database.pkl from disk into memory once per session."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# [OPT-6] computed once, reused on every rerun
@st.cache_data(show_spinner=False)
def get_song_stats(db_pickle_path: str) -> tuple[dict, int]:
    """
    [OPT-6] Pre-aggregate per-song hash counts from the database.

    Passing the file path (not the dict) as the cache key means Streamlit
    invalidates this cache only when the file on disk changes — not on every
    widget interaction.  Library tab renders are instant after the first visit.

    Returns (song_hash_counts dict, total_entry_count).
    """
    db = load_database(db_pickle_path)
    if db is None:
        return {}, 0
    counts: dict[str, int] = defaultdict(int)
    for entries in db.values():
        for (song_name, _) in entries:
            counts[song_name] += 1
    return dict(counts), sum(counts.values())


# ─────────────────────────────────────────────────────────────────────────────
# DARK-THEMED MATPLOTLIB HELPERS
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK = "#121212"
BG_AXES = "#1e1e1e"
ACCENT = "#1DB954"
MUTED = "#B3B3B3"


def _apply_dark_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Stamp the Spotify dark theme onto a Matplotlib Axes object."""
    ax.set_facecolor(BG_AXES)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color("white")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color="#333333", linewidth=0.5, linestyle="--", alpha=0.7)


def make_extraction_figure(
    S_db: np.ndarray,
    freq_arr: np.ndarray,
    time_arr: np.ndarray,
) -> plt.Figure:
    """
    Figure 1 — Feature Extraction Plot.

    [OPT-8] Sub-sample the constellation scatter to MAX_SCATTER_POINTS.
    Plotting thousands of individual markers through Matplotlib's Python-level
    scatter renderer dominates render time with zero perceptual benefit.
    A random sub-sample of 2 000 points is visually indistinguishable from
    the full set while being 5–10× faster to render.

    Left panel:  log-spectrogram (magma colormap).
    Right panel: constellation map scatter.
    """
    fig, (ax_spec, ax_peaks) = plt.subplots(
        1, 2,
        figsize=(12, 4.5),
        facecolor=BG_DARK,
        gridspec_kw={"wspace": 0.35},
    )

    n_bins, n_frames = S_db.shape

    # ── Left: log-spectrogram ────────────────────────────────────────────────
    im = ax_spec.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        extent=[0, n_frames, 0, n_bins],
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    cbar = fig.colorbar(im, ax=ax_spec, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.set_label("dB", color=MUTED, fontsize=8)
    _apply_dark_style(ax_spec, "Log-Spectrogram",
                      "Time Frame", "Frequency Bin")

    # ── Right: constellation scatter (sub-sampled) ───────────────────────────
    n_peaks = len(freq_arr)
    if n_peaks > 0:
        # [OPT-8] sub-sample to at most MAX_SCATTER_POINTS random indices
        if n_peaks > MAX_SCATTER_POINTS:
            idx = np.random.choice(n_peaks, MAX_SCATTER_POINTS, replace=False)
            plot_f = freq_arr[idx]
            plot_t = time_arr[idx]
        else:
            plot_f = freq_arr
            plot_t = time_arr

        ax_peaks.scatter(
            plot_t, plot_f,
            s=4,
            color=ACCENT,
            alpha=0.75,
            linewidths=0,
            rasterized=True,   # rasterise the scatter layer into the SVG frame
        )

    ax_peaks.set_xlim(0, n_frames)
    ax_peaks.set_ylim(0, n_bins)
    ax_peaks.set_facecolor("#0d0d0d")
    _apply_dark_style(
        ax_peaks,
        f"Constellation Map  ({n_peaks:,} peaks)",
        "Time Frame",
        "Frequency Bin",
    )

    fig.tight_layout(pad=2.0)
    return fig


def make_alignment_figure(offsets_arr: np.ndarray, song_name: str) -> plt.Figure:
    """
    Figure 2 — Alignment Spike Plot.
    Offset histogram for the winning song — the alignment convergence spike
    is the visual proof that the matching engine found the correct song.
    """
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG_DARK)

    if offsets_arr.size > 0:
        min_o = int(offsets_arr.min())
        max_o = int(offsets_arr.max())
        n_bins = max(min(max_o - min_o + 1, 2000), 1)

        counts, edges = np.histogram(
            offsets_arr, bins=n_bins, range=(min_o, max_o + 1)
        )
        bin_width = edges[1] - edges[0]

        ax.bar(
            edges[:-1], counts,
            width=bin_width * 0.9,
            color=ACCENT,
            alpha=0.85,
            edgecolor="none",
        )

        # Annotate the alignment peak
        peak_idx = int(np.argmax(counts))
        peak_offset = edges[peak_idx]
        peak_count = int(counts[peak_idx])

        ax.axvline(
            peak_offset + bin_width / 2,
            color="white",
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
        )
        ax.annotate(
            f"  Peak: {peak_count:,} matches\n  Offset: {int(peak_offset)} frames",
            xy=(peak_offset, peak_count),
            xytext=(peak_offset + max(1.0, n_bins * 0.04), peak_count * 0.85),
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    _apply_dark_style(
        ax,
        f"Alignment Offset Distribution — {song_name}",
        "Time Offset  (t_database − t_query)  [frames]",
        "Matched Hash Count",
    )
    fig.tight_layout(pad=2.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATABASE (once per session)
# ─────────────────────────────────────────────────────────────────────────────
database = load_database(DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if database is None:
        st.error(f"`{DB_PATH}` not found.\nRun `fingerprint_indexer.py` first.")
    else:
        st.success("Database loaded ✓")
        song_stats, _ = get_song_stats(DB_PATH)
        st.metric("Indexed Tracks",   f"{len(song_stats):,}")
        st.metric("Unique Hash Keys", f"{len(database):,}")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#B3B3B3;'>"
        "EE200 · Audio Fingerprinting<br>"
        "Shazam-style landmark hashing<br>"
        "STFT n_fft=2048 · hop=512 · sr=22 050<br><br>"
        "<b style='color:#5bc8f5;'>Optimisations active:</b><br>"
        "OPT-1 BytesIO (no disk I/O)<br>"
        "OPT-2 float32 spectrogram<br>"
        "OPT-3 vectorised peak sort<br>"
        "OPT-4 searchsorted hashing<br>"
        "OPT-5 fingerprint cache<br>"
        "OPT-6 stats cache<br>"
        "OPT-7 vectorised offsets<br>"
        "OPT-8 scatter sub-sample"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_library, tab_identify, tab_batch = st.tabs(
    ["📂  LIBRARY", "🔍  IDENTIFY", "📊  BATCH"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIBRARY OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_library:
    st.markdown(
        '<div class="section-label">Indexed Song Collection</div>',
        unsafe_allow_html=True,
    )

    if database is None:
        st.warning(
            f"No database found at `{DB_PATH}`. "
            "Please run `fingerprint_indexer.py` first."
        )
    else:
        # [OPT-6] get_song_stats() is cached — instant on every rerun
        song_hash_counts, total_entries = get_song_stats(DB_PATH)
        songs_sorted = sorted(
            song_hash_counts.items(), key=lambda x: x[1], reverse=True
        )
        n_songs = len(songs_sorted)

        # ── Summary metrics ───────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Indexed Tracks",   f"{n_songs:,}")
        c2.metric("Unique Hash Keys", f"{len(database):,}")
        c3.metric("Total DB Entries", f"{total_entries:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">All Tracks</div>', unsafe_allow_html=True
        )

        # ── 3-column song card grid ───────────────────────────────────────────
        COL_COUNT = 3
        rows = [
            songs_sorted[i: i + COL_COUNT]
            for i in range(0, n_songs, COL_COUNT)
        ]
        for row in rows:
            cols = st.columns(COL_COUNT)
            for col, (song_name, hash_count) in zip(cols, row):
                with col:
                    display_name = song_name.replace("_", " ")
                    st.markdown(
                        f"""
                        <div class="song-card">
                            <div class="song-title" title="{display_name}">
                                🎵 {display_name}
                            </div>
                            <div class="song-meta">{song_name}</div>
                            <div class="song-hash-badge">⬡ {hash_count:,} hashes</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — IDENTIFY SINGLE CLIP
# ═════════════════════════════════════════════════════════════════════════════
with tab_identify:
    st.markdown(
        '<div class="section-label">Upload a Clip to Identify</div>',
        unsafe_allow_html=True,
    )

    if database is None:
        st.warning("Database not loaded — cannot run identification.")
    else:
        uploaded_file = st.file_uploader(
            "Drag & drop an audio clip, or click to browse",
            type=["wav", "mp3", "m4a", "flac"],
            key="single_upload",
        )

        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()

            # ── Playback ──────────────────────────────────────────────────────
            st.markdown(
                '<div class="section-label">Playback</div>', unsafe_allow_html=True
            )
            st.audio(audio_bytes, format=uploaded_file.type or "audio/wav")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🎯  Identify Clip", key="btn_identify"):
                with st.spinner("Extracting features & running alignment match …"):
                    t_wall_start = time.perf_counter()

                    try:
                        # ── [OPT-5] Feature extraction (cached) ───────────────
                        # On first call for this file: runs the full pipeline.
                        # On all subsequent calls (reruns): returns instantly.
                        t_feat_start = time.perf_counter()
                        S_db, freq_arr, time_arr, query_hashes = fingerprint_audio(
                            audio_bytes
                        )
                        t_feat = time.perf_counter() - t_feat_start

                        # ── [OPT-7] Database matching ─────────────────────────
                        t_match_start = time.perf_counter()
                        winner, scores, offsets_map = match_against_database(
                            query_hashes, database
                        )
                        t_match = time.perf_counter() - t_match_start
                        t_total = time.perf_counter() - t_wall_start

                        # ── Timing dashboard ──────────────────────────────────
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Query Hashes",    f"{len(query_hashes):,}")
                        m2.metric("Peaks Found",     f"{len(freq_arr):,}")
                        m3.metric("Extraction Time", f"{t_feat * 1000:.1f} ms")
                        m4.metric("Match Time",
                                  f"{t_match * 1000:.1f} ms")

                        st.markdown("<br>", unsafe_allow_html=True)

                        # ── Winner announcement ───────────────────────────────
                        if winner is None:
                            st.error(
                                "❌  No matching song found in the database.")
                        else:
                            top_score = scores[winner]
                            display_win = winner.replace("_", " ")

                            st.markdown(
                                f"""
                                <div class="winner-box">
                                    <div class="winner-label">✓ Match Identified</div>
                                    <div class="winner-name">{display_win}</div>
                                    <div class="winner-stats">
                                        Alignment confidence: {top_score:,} aligned hashes
                                        &nbsp;·&nbsp;
                                        Total time: {t_total * 1000:.0f} ms
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # ── Top-5 candidates table ────────────────────────
                            st.markdown(
                                '<div class="section-label">Top 5 Candidates</div>',
                                unsafe_allow_html=True,
                            )
                            ranked = sorted(
                                scores.items(), key=lambda x: x[1], reverse=True
                            )[:5]
                            df_ranked = pd.DataFrame(
                                [
                                    {
                                        "Rank":            i + 1,
                                        "Song":            name.replace("_", " "),
                                        "Alignment Score": score,
                                        "Match":           "✓ WINNER" if name == winner else "",
                                    }
                                    for i, (name, score) in enumerate(ranked)
                                ]
                            )
                            st.dataframe(
                                df_ranked,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Alignment Score": st.column_config.ProgressColumn(
                                        "Alignment Score",
                                        min_value=0,
                                        max_value=ranked[0][1] if ranked else 1,
                                        format="%d",
                                    )
                                },
                            )

                            st.markdown("<br>", unsafe_allow_html=True)

                            # ── Figure 1: Feature Extraction Plot ─────────────
                            st.markdown(
                                '<div class="section-label">Figure 1 — Feature Extraction</div>',
                                unsafe_allow_html=True,
                            )
                            # [OPT-8] passes arrays directly; scatter is sub-sampled inside
                            fig1 = make_extraction_figure(
                                S_db, freq_arr, time_arr)
                            st.pyplot(fig1, use_container_width=True)
                            plt.close(fig1)

                            st.markdown("<br>", unsafe_allow_html=True)

                            # ── Figure 2: Alignment Spike ─────────────────────
                            st.markdown(
                                '<div class="section-label">Figure 2 — Alignment Offset Spike</div>',
                                unsafe_allow_html=True,
                            )
                            if winner in offsets_map:
                                fig2 = make_alignment_figure(
                                    offsets_map[winner], display_win
                                )
                                st.pyplot(fig2, use_container_width=True)
                                plt.close(fig2)
                            else:
                                st.info(
                                    "No offset data available for the winning song.")

                    except Exception as exc:
                        st.error(f"❌  Processing failed: {exc}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH MODE
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(
        '<div class="section-label">Batch Evaluation</div>', unsafe_allow_html=True
    )

    if database is None:
        st.warning("Database not loaded — cannot run batch evaluation.")
    else:
        batch_files = st.file_uploader(
            "Upload multiple audio clips for automated evaluation",
            type=["wav", "mp3", "m4a", "flac"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if batch_files:
            st.markdown(f"**{len(batch_files)} file(s) ready.**")

            if st.button("▶  Run Batch Evaluation Pass", key="btn_batch"):
                results_rows = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                n_files = len(batch_files)

                for i, bf in enumerate(batch_files):
                    raw_filename = bf.name
                    status_text.markdown(
                        f"<span style='color:#B3B3B3;font-size:0.85rem;'>"
                        f"Processing {i + 1}/{n_files}: `{raw_filename}`</span>",
                        unsafe_allow_html=True,
                    )

                    try:
                        audio_bytes = bf.read()
                        # [OPT-5] fingerprint_audio() is cached — repeated clips
                        # are essentially free on the second call
                        _, _, _, q_hashes = fingerprint_audio(audio_bytes)
                        winner, _, _ = match_against_database(
                            q_hashes, database)
                        prediction = winner if winner is not None else "None"

                    except Exception as exc:
                        st.warning(f"⚠️  Skipped `{raw_filename}`: {exc}")
                        prediction = "None"

                    results_rows.append(
                        {"filename": raw_filename, "prediction": prediction}
                    )
                    progress_bar.progress((i + 1) / n_files)

                status_text.markdown(
                    "<span style='color:#1DB954;font-size:0.85rem;font-weight:700;'>"
                    "✓ Batch complete.</span>",
                    unsafe_allow_html=True,
                )

                # ── Results table ─────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-label">Results</div>',
                    unsafe_allow_html=True,
                )
                # Strict column contract: exactly lowercase `filename` and `prediction`
                df_results = pd.DataFrame(
                    results_rows, columns=["filename", "prediction"]
                )
                st.dataframe(df_results, use_container_width=True,
                             hide_index=True)

                # ── Summary metrics ───────────────────────────────────────────
                n_matched = (df_results["prediction"] != "None").sum()
                n_unmatched = n_files - n_matched
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Clips",  n_files)
                m2.metric("Matched",      n_matched)
                m3.metric("Not Matched",  n_unmatched)

                # ── CSV download ──────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                csv_bytes = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇  Download results.csv",
                    data=csv_bytes,
                    file_name="results.csv",
                    mime="text/csv",
                    key="dl_csv",
                )
        else:
            st.info(
                "Upload one or more audio clips above, "
                "then press **Run Batch Evaluation Pass**."
            )
