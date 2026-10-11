"""Evaluation behind the published numbers.

=============  ===============================================================
module         what it scores
=============  ===============================================================
``ensemble``   checkpoint ensembles on the lidar chips, paired chip by chip
``external``   the model vs Copernicus, FABDEM and GEDTM30 on the same pixels
``blocks``     ``external`` paired differences resampled by split block
``icesat2``    DTMs vs ICESat-2 ground segments where no lidar exists
``stats``      chip, block and overpass bootstraps shared by the above
=============  ===============================================================

Each tool runs as ``python -m canopy2ground.evaluation.<module>``.
"""
