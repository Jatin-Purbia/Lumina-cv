# Manual data checklist

No downloader is implemented and no dataset has been fetched.

1. Follow [DATA](../docs/DATA.md) primary author links for SICE part 1, the
   Zero-DCE training archive, original LOL, and unpaired image sets.
2. Inspect and privately record archive/source terms, source URL, date and hash.
3. Extract outside tracked files; normalize to the documented data/ folders.
4. Select low exposures, split by scene/session and save private manifests.
5. Set the `paths` section in the experiment config (`configs/baseline.py`); keep LOL evaluation disjoint.
6. Before M1, implement decoder/schema stubs and audit missing/duplicate files.
