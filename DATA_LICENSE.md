# Data license

The knowledge graph and pipeline outputs in `data/` are released under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The code is released under the MIT License (see `LICENSE`).

The released data builds on third-party sources, which keep their own terms:

| Source | Used for | License |
| --- | --- | --- |
| [OpenAlex](https://openalex.org/) | Paper metadata (`data/papers/`, Paper/Author/Venue nodes) | CC0 |
| [ESGF](https://esgf.github.io/) search metadata | KG backbone (Source, Experiment, Variable, ... nodes) | ESGF / CMIP6 terms of use |
| [WCRP CMIP6_CVs](https://github.com/WCRP-CMIP/CMIP6_CVs) | `kg_create/other_sources/CMIP6_source_id.json`, model components | CC BY 4.0 |
| [GeoNames](https://www.geonames.org/) | `kg_create/geolocation/` | CC BY 4.0 |
| [NASA GCMD Keywords](https://www.earthdata.nasa.gov/data/tools/gcmd-keyword-viewer) | `kg_create/gcmd/` | Public domain (NASA) |

Each extracted triplet stores the short evidence paragraph it was extracted from,
quoted from the source publication for verification. Full texts and PDFs are not
redistributed.
