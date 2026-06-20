#!/usr/bin/env python3
"""
Audio Fingerprinting Indexer — Shazam-style Landmark Hashing
=============================================================
Scans a directory of audio tracks, extracts constellation map peaks from their
log-spectrograms, generates combinatorial anchor-target hash pairs, and serializes
the resulting index to disk as `song_database.pkl`.

Architecture
------------
  get_spectrogram()      — Load audio & compute log-magnitude STFT
  extract_peaks()        — Detect local maxima (constellation map)
  generate_hashes()      — Pair anchors with target-zone peaks → hash keys
  build_index_database() — Orchestrate the full pipeline over a directory

Usage
-----
  python fingerprint_indexer.py --audio_dir ./tracks --db_out song_database.pkl
"""

import os
import pickle
import argparse
import warnings
from collections import defaultdict

import numpy as np
import librosa
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------------------
# Global DSP / hashing constants  (tweak here; no magic numbers in functions)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22_050   # Hz — uniform downsample target
N_FFT = 2048     # STFT window length (samples)
HOP_LENGTH = 512      # STFT hop / overlap (samples)

# Side length of the 2-D max-filter kernel (bins × frames)
PEAK_NEIGHBORHOOD = 30
# Peaks below this dB level (rel. to max) are discarded
MIN_DB_THRESHOLD = -20.0

# Target-zone look-ahead window (in STFT frames)
TZ_OFFSET_MIN = 5        # Minimum forward offset from the anchor
# Maximum forward offset from the anchor (lookahead horizon)
TZ_OFFSET_MAX = 50

# Supported audio container formats
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# Step 1 — Audio loading & log-spectrogram computation
# ---------------------------------------------------------------------------

def get_spectrogram(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample to SAMPLE_RATE, convert to mono, and compute
    a log-magnitude (dB) spectrogram via STFT.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the audio file.

    Returns
    -------
    S_db : np.ndarray, shape (n_fft//2 + 1, n_frames)
        Log-magnitude spectrogram in dB, normalised against the peak magnitude.
    sr   : int
        Actual sample rate used (always equals SAMPLE_RATE).
    """
    # librosa.load automatically resamples and mixes down to mono
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Short-Time Fourier Transform → complex spectrogram
    stft_matrix = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Convert complex STFT to amplitude, then to dB scale.
    # ref=np.max normalises so the loudest bin is 0 dB; everything else is negative.
    amplitude = np.abs(stft_matrix)
    S_db = librosa.amplitude_to_db(amplitude, ref=np.max)

    return S_db, sr


# ---------------------------------------------------------------------------
# Step 2 — Constellation map: local-maxima detection
# ---------------------------------------------------------------------------

def extract_peaks(
    S_db: np.ndarray,
    neighborhood: int = PEAK_NEIGHBORHOOD,
    db_threshold: float = MIN_DB_THRESHOLD,
) -> list[tuple[int, int]]:
    """
    Identify spectral peak points that are simultaneously:
      (a) the maximum value within a (neighborhood × neighborhood) 2-D window, AND
      (b) above the minimum dB threshold (filters silence / noise floor).

    The result is a "constellation map" — a sparse set of salient time-frequency
    landmarks that robustly survive background noise and moderate distortion.

    Parameters
    ----------
    S_db         : np.ndarray   Log-magnitude spectrogram (freq bins × time frames).
    neighborhood : int          Side length of the 2-D maximum-filter kernel.
    db_threshold : float        Minimum dB value for a valid peak.

    Returns
    -------
    peaks : list of (freq_bin: int, time_frame: int)
        Sorted by time frame (chronological order).
    """
    # Apply a 2-D sliding maximum filter across the spectrogram.
    # A point IS a local maximum iff its value equals the filtered output at that point.
    kernel_shape = (neighborhood, neighborhood)
    local_max_map = maximum_filter(
        S_db, size=kernel_shape, mode="constant", cval=0.0)
    is_local_max = S_db == local_max_map

    # Suppress peaks that lie on or below the noise-floor threshold
    above_threshold = S_db > db_threshold

    # Boolean mask: True only where both conditions are satisfied
    peak_mask = is_local_max & above_threshold

    # Extract (freq_bin, time_frame) index pairs
    freq_indices, time_indices = np.where(peak_mask)

    # Zip and sort chronologically by time frame for the forward-looking pairing step
    peaks = sorted(zip(freq_indices.tolist(),
                   time_indices.tolist()), key=lambda p: p[1])

    return peaks


# ---------------------------------------------------------------------------
# Step 3 — Combinatorial anchor→target hashing (the Shazam core)
# ---------------------------------------------------------------------------

def generate_hashes(
    peaks: list[tuple[int, int]],
    tz_offset_min: int = TZ_OFFSET_MIN,
    tz_offset_max: int = TZ_OFFSET_MAX,
) -> list[tuple[tuple[int, int, int], int]]:
    """
    Pair every constellation peak (anchor) with every subsequent peak that falls
    inside its forward-looking "Target Zone", producing a set of time-invariant
    hash keys.

    Hash key   : (f1, f2, delta_t)
        f1      — frequency bin of the anchor peak
        f2      — frequency bin of the target peak
        delta_t — time-frame difference (t2 - t1); encodes relative timing, not absolute

    Because delta_t is a *difference*, the same musical passage produces the same hash
    regardless of where in the recording it occurs — enabling offset-aligned matching
    during the query/lookup phase.

    Parameters
    ----------
    peaks         : list[(freq_bin, time_frame)]   Sorted constellation points.
    tz_offset_min : int   Minimum forward frame offset to begin the target zone.
    tz_offset_max : int   Maximum forward frame offset (lookahead horizon).

    Returns
    -------
    hashes : list of ( (f1, f2, delta_t), t1 )
        Each entry pairs its hash key with the anchor's absolute frame position t1
        (stored in the DB so we can recover timing during matching).
    """
    hashes = []
    n_peaks = len(peaks)

    for i, (f1, t1) in enumerate(peaks):
        # Search forward through peaks whose time frame falls within the target zone
        for j in range(i + 1, n_peaks):
            f2, t2 = peaks[j]
            delta_t = t2 - t1

            # Skip peaks not yet inside the minimum offset
            if delta_t < tz_offset_min:
                continue

            # Stop once we exceed the maximum lookahead horizon
            if delta_t > tz_offset_max:
                break  # peaks are time-sorted, so no further pair can be in range

            # Emit the (hash_key, anchor_time) pair
            hash_key = (f1, f2, delta_t)
            hashes.append((hash_key, t1))

    return hashes


# ---------------------------------------------------------------------------
# Step 4 — Directory scan, pipeline orchestration, and DB serialisation
# ---------------------------------------------------------------------------

def build_index_database(audio_dir: str, db_output_path: str) -> None:
    """
    Walk `audio_dir`, fingerprint every supported audio file, and accumulate all
    hashes into an inverted index dictionary:

        { (f1, f2, delta_t): [(song_name, t1), (song_name, t1), ...] }

    The final dictionary is serialised to disk via pickle.

    Parameters
    ----------
    audio_dir      : str   Path to the directory containing audio tracks.
    db_output_path : str   Destination path for the serialised `song_database.pkl`.
    """
    # Inverted index: hash_key → list of (song_name, anchor_t1) tuples
    # defaultdict(list) means we never have to initialise a key before appending
    database: dict[tuple, list] = defaultdict(list)

    # Collect all audio files in the target directory (non-recursive)
    audio_files = [
        f for f in os.listdir(audio_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        print(f"[WARNING] No supported audio files found in '{audio_dir}'.")
        print(
            f"          Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        return

    total_files = len(audio_files)
    print(f"\n{'='*60}")
    print(f"  Audio Fingerprint Indexer")
    print(f"{'='*60}")
    print(f"  Source directory : {os.path.abspath(audio_dir)}")
    print(f"  Output database  : {os.path.abspath(db_output_path)}")
    print(f"  Tracks found     : {total_files}")
    print(f"{'='*60}\n")

    total_hashes = 0

    for idx, filename in enumerate(sorted(audio_files), start=1):
        # Strip the file extension to obtain a clean song label
        song_name = os.path.splitext(filename)[0]
        file_path = os.path.join(audio_dir, filename)

        print(f"[{idx:>3}/{total_files}] Processing : '{song_name}'")

        try:
            # ── Stage 1: Load & compute log-spectrogram ──────────────────────
            S_db, sr = get_spectrogram(file_path)
            n_bins, n_frames = S_db.shape
            print(
                f"           Spectrogram  : {n_bins} freq bins × {n_frames} frames")

            # ── Stage 2: Detect constellation peaks ──────────────────────────
            peaks = extract_peaks(S_db)
            print(f"           Peaks found  : {len(peaks)}")

            if len(peaks) < 2:
                print(
                    f"           [SKIP] Too few peaks to generate hashes — file may be silent.\n")
                continue

            # ── Stage 3: Generate combinatorial hashes ────────────────────────
            hashes = generate_hashes(peaks)
            song_hash_count = len(hashes)
            print(f"           Hashes built : {song_hash_count:,}")

            # ── Stage 4: Insert into the inverted index ───────────────────────
            for hash_key, t1 in hashes:
                database[hash_key].append((song_name, t1))

            total_hashes += song_hash_count
            print(f"           ✓ Indexed successfully.\n")

        except Exception as exc:
            # Isolate failures so one corrupted file cannot abort the entire run
            print(
                f"           [ERROR] Failed to process '{filename}': {exc}\n")
            continue

    # ── Serialise the completed database to disk ──────────────────────────────
    print(f"{'='*60}")
    print(f"  Indexing complete.")
    print(f"  Total unique hash keys : {len(database):,}")
    print(f"  Total hash entries     : {total_hashes:,}")
    print(f"  Saving database → '{db_output_path}' …", end=" ", flush=True)

    with open(db_output_path, "wb") as fh:
        pickle.dump(dict(database), fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(db_output_path) / 1024
    print(f"done  ({size_kb:.1f} KB)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Shazam-style audio fingerprint index from a directory of tracks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./tracks",
        help="Directory containing audio files to index.",
    )
    parser.add_argument(
        "--db_out",
        type=str,
        default="song_database.pkl",
        help="Output path for the serialised fingerprint database.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Suppress librosa/audioread deprecation chatter for cleaner console output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

   # Hardcoded paths pointing to your exact folder structure
    my_audio_folder = "./Course_Songs_Library"
    my_database_output = "./song_database.pkl"

    # Verify the folder can be found locally
    if not os.path.isdir(my_audio_folder):
        raise SystemExit(
            f"[FATAL] Audio directory not found: '{my_audio_folder}'")

    # Run the database compiler pass
    build_index_database(
        audio_dir=my_audio_folder,
        db_output_path=my_database_output,
    )
