"""Compose Nanjing + Changsha triptychs into one 2-row main-text figure ->
submission/figures/Figure4_maps_maintext.pdf
Pure image composition; does not re-run the LOCO pipeline.

2026-09-12: briefly trimmed to Nanjing-only for page budget, then reverted
the same day -- the two-city figure is more convincing (shows the boundary-
clustering pattern isn't a one-city artefact) and the rationale for which two
cities is now stated explicitly in the caption/prose instead of only living
in this script's comment. See
memory/loco-trim-and-manuscript-audit-2026-09-12.md."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

root = Path("/Users/alex/Projects/gee-project")
src = root / "data" / "analysis_outputs" / "correction_maps"  # scripts/make_correction_maps.py output
imgs = [("Nanjing",  src / "Fig_diag_map_Nanjing.png"),
        ("Changsha", src / "Fig_diag_map_Changsha.png")]

fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.9))
for ax, (city, path) in zip(axes, imgs):
    ax.imshow(mpimg.imread(path))
    ax.set_axis_off()
    ax.text(-0.012, 0.5, city, transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=12, fontweight="bold")
fig.subplots_adjust(left=0.03, right=0.995, top=0.995, bottom=0.005, hspace=0.04)

out = root / "submission/figures/Figure4_maps_maintext.pdf"
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix(".png"), dpi=200)
print("wrote", out)
print("wrote", out.with_suffix(".png"))
