# Final source availability audit — 2026-10-11

These official-page checks supplement the 2026-10-10 paper snapshot. They do not represent model execution or data-access approval.

## FLAME

The [official project](https://flame.is.tue.mpg.de/) offers **FLAME2023 Open under CC-BY-4.0** and links an [expression converter](https://github.com/Suraj-Bhor/flame_to_flame_open_converter). Older versions have different terms. Check converted geometry before changing RealDenseFace's expected model asset. This corrects an availability assumption; it does not alter the original FLAME method.

The old MPI preprint endpoint returned HTML during acquisition. The original archived PDF is a title-verified 17-page mirror copy. The official site now also points to [a working author PDF endpoint](https://download.is.tue.mpg.de/flame/flame_paper.pdf); web extraction refused its approximately 40MB payload. The local verified copy remains unchanged and its SHA is retained.

## NeRSemble

The correct [author project page](https://tobias-kirschstein.github.io/nersemble/) and [v2 data repository](https://github.com/tobias-kirschstein/nersemble-data) replace an unverified guessed project URL. The current [benchmark](https://kaldir.vc.cit.tum.de/nersemble_benchmark/) confirms a 391-image single-view reconstruction task with separate posed/neutral geometry evaluation. It also hosts separate dynamic NVS and monocular FLAME-video tasks. Keep their inputs and test sets distinct.

## Tool versus model environment

The research utilities ran using Python3.14, requests2.33.1 and PyMuPDF1.28.2 on Windows. Those packages are for PDF acquisition/inspection, **not** an environment lock for RealDenseFace or GPU rendering. Pin the model environment during Phase0. The current nvdiffrast documentation and the old rendering paper may describe different supported library/toolchain versions.

The local catalog validator checks files, PDF identities, hashes and local links. It does not test every external URL, checkpoint, permission, GPU kernel or advertised dependency.
