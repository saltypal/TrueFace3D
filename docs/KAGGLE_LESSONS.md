# Kaggle lessons (learned the hard way, NoW benchmark runs, Nov 2026)

## Environment and builds
1. **Pin the image.** Set `docker_image_pinning_type: "original"` in every `kernel-metadata.json`. A new image
   breaks the cached CUDA wheels.
2. **Precompile CUDA once, cache as wheels.** nvdiffrast, the FLAME solver extension and PyTorch3D take a long
   time to build. Build them once on the exact image (torch 2.11 + cu130, Python 3.13, T4), save the wheels in a
   dataset (`satyapaladugu/mesha-output/wheels`), attach it, and install from the wheel. Setup should report
   "installed from wheel"; if it says "built", the cache was missed.
3. **CUDA ≥ 13 build flags:** `NVCC_APPEND_FLAGS=-static-global-template-stub=false` and `FORCE_CUDA=1`.
4. **Old packages need patching on disk, not in memory.** chumpy: fetch the sdist, then fix the numpy import and
   `getargspec → getfullargspec` in the installed files. Subprocesses import the file, so an in-memory patch is
   not enough.
5. **Write patch code with byte-safe edits.** Escapes in a heredoc once wrote control characters into
   `setup.py`. Compile-check after every patch.
6. **List all pip deps up front** (yacs, iopath, fvcore, kornia, face_alignment, pyshtools). Each missing one
   costs a full rerun.

## Data and weights
7. **Kaggle auto-extracts archives** in datasets. Find weights by file name (`rglob`), never by a fixed path.
8. **git-lfs files arrive as pointer files** unless git-lfs is installed (`apt`). Check file sizes.
9. **Download the exact documented checkpoint.** The wrong MAE file ("visualize" instead of "pretrain") loaded
   without an error and gave wrong results.
10. **Find inputs with `glob('/kaggle/input/**/...')`.** The mount path changes (`/kaggle/input/datasets/<user>/...`).

## Running
11. **Assert after every expensive step:** builds OK → data found → one smoke job scored → launch the rest →
    enough results → zip. A failed assertion costs minutes; a silent failure costs a whole GPU session.
12. **Use both T4s:** split jobs across GPUs with one subprocess per GPU (`src/kaggle/runner.py`).
13. **Print progress every ~2 minutes** plus per-job start/done lines, so a stuck run is visible in the log.
14. **Keep a scoring error next to the result instead of dropping the result.** One bad API call
    (`Image.Resampling.AREA` doesn't exist; use `BOX`) once threw away every clean appearance result.
15. **Count the jobs before launching** (show the math). Remove sets you don't need; chunk so one session
    finishes well under the 12 h limit.
16. **Max 2 GPU sessions per account, and the CLI cannot cancel runs.** Cancel old versions on the website before
    pushing a new one. Two accounts give two parallel lanes.
17. **Check that coordinate frames match** before swapping data between runs (oracle bug: per-vertex
    coordinates are in input-image pixels and must be mapped between face boxes).

## Outputs
18. **The last cell writes one zip** of the output folder (`shutil.make_archive`), so there is a single file to
    download.
19. **For big outputs, save the output as a dataset** and download that (`kaggle datasets download`). A separate
    "zip it" notebook can fail and return only a log.
20. **Zip only what you need.** The 6.3 GB output was mostly a repo copy, build files and weights. Results
    (`*.json`, `neutral.obj`, `priors.npz`) were the useful part. Exclude `.git`, builds and weights from the zip.
21. **Check local disk space before downloading**, and extract selectively.
22. **Record the git commit, kernel version, seeds and config** in the output so every result can be traced.
