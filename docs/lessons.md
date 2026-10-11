# What mattered

These are the decisions that moved the result, roughly in order of size. The numbers are on the validation chips
unless noted. The early ones were measured on the smaller validation set used at the time.

**1. Choice of prior DEM.** Swapping SRTM for Copernicus GLO-30 as the input DEM cut the error of the *untrained*
prior from 3.24 to 1.64 m. Copernicus with no model at all beat the best trained model on SRTM. Every later
architectural gain was a fraction of that.

**2. Priors derived from the label.** An early synthetic prior was made by contouring the lidar and
re-interpolating it. It produced impressive scores and cannot exist at deployment time. The leak is not about how
much detail the prior carries. By `|r(residual, highpass(prior))|` the contour-derived prior (0.18) looked no
leakier than Copernicus (0.22). The leak is that it is a function of the label, so a model can learn to invert that
function. The rule adopted was to use only priors that exist, independently measured, over the deployment area.

**3. Testing the controls.** The decoy that measures how much skill comes from the imagery swaps each chip's
photographs for another chip's. The first version permuted inside a micro-batch of four. A random permutation has
one fixed point on average, so about a quarter of the "swapped" chips kept their own imagery. The control was then
under-reporting exactly the failure it exists to catch. Each chip now pairs with one half the dataset away, and a
test checks that no chip is ever paired with itself
([`tests/test_metrics_and_speed.py`](../tests/test_metrics_and_speed.py)).

**4. Loss on elevation vs loss per wavelength band.** Almost all of a chip's variance is hillside shape, and an
L1-optimal model under uncertainty damps every band by its correlation with the truth. Adding a per-octave
correlation loss raised ridge-band correlation from +0.24 to +0.40 and the 2–4 m band roughly fivefold. The
amplitude term then traded a little MAE for relief that is actually present.

**5. Checking the output spectrum, not only the error.** The first published map looked fine in error metrics. Its
hillshade spectrum had sharp peaks at exactly 2.0 m and 4.0 m along the map axes, ×400–2,500 above background. The
lidar and every input lacked them. The cause was architectural: a stride-4 patch embedding, an empty H/2 skip and
nearest-neighbour upsampling. Shift-TTA cancelled it in the deployed ensemble, and the bilinear decoder with a
learned skip removed it at the source.

**6. Longer training before larger encoders.** Tripling the schedule (22 → 66 epochs) cut median error by 0.12 m
and transferred to the withheld region. Deeper encoders (MiT-B3, MiT-B5) at the same learning rate gave no reliable
gain. At a 0.3× encoder learning rate, MiT-B5 raised imagery-attributable correlation by 0.012 with no change in
MAE. It was kept because adding two MiT-B5 members cut ensemble MAE by 0.020 m; a MiT-B5 model did not beat MiT-B2
on the withheld region.

**7. A speedup that came with better numbers.** `torch.compile` gave a 1.7–2.0× step speedup and, apparently,
better training skill. Both came from skipped steps: every compiled run had non-finite losses that the GradScaler
silently dropped. The speedup was withdrawn, and training now counts skipped steps and aborts when they exceed 5%
of an epoch.

**8. Coastline and absolute elevation as inputs.** The 30 m DEM's coastline is tens of metres off in places, and
the model had learned to invent land in coves. Giving it the County coastline and absolute elevation as inputs,
plus a sea-level loss, cut the false-land rate on sea pixels from 8.4% (prior) to 2.7–3.3%. Accuracy on land did
not suffer.

Negative results are kept in [results.md §7](results.md#7-design-decisions-that-were-tested). They include
geometric augmentation (it hurt; a likely reason is that rotating a chip moves the sun relative to the terrain), a
canopy-bias loss (it traded bias for error about one for one), larger batches, and a mixed result for 1024 m
context windows.
