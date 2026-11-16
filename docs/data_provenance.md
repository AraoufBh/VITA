# Data provenance

## Time series (data tier B)

- Source: ENTSO-E Transparency Platform, Spain: hourly generation per production type and day-ahead prices, 2015 to
  2018, in the public compilation `energy_dataset.csv` distributed with the open-source Darts time-series library
  (`https://raw.githubusercontent.com/unit8co/darts/master/datasets/energy_dataset.csv`).
- File: 6,062,625 bytes, SHA-256 `6ccdba32b23e4ebb7fa4dc31b93848ea778e5c918007a5ebbb6fa2cc993886c5`, 35,064 hourly
  rows, naive UTC timestamps from 2014-12-31 23:00 to 2018-12-31 22:00.
- Columns used: `generation solar`, `generation wind onshore`, `price day ahead`.
- Processing (`src/vita/data.py`): timestamps shifted by +1 h (fixed UTC+1), 29 February removed, 8760 hours per year;
  gaps of at most 3 hours filled linearly, longer gaps with the same hour one week earlier (PV and wind: 16 hours in
  2015, 1 hour in 2016 and 1 hour in 2018; prices: none); capacity factor proxy CF_t = gen_t / Q_y(0.999) clipped to
  [0, 1].
- Limitation: the proxy is a normalized national generation profile, not a site capacity factor. It absorbs
  capacity additions within a year and curtailment.
- The ENTSO-E Transparency Platform data are published under the platform's terms of use; the Darts library is
  distributed under the Apache 2.0 license. The raw file is not redistributed in this repository; `make data`
  downloads it and checks the checksum.

## Data tier A (not used in the published run)

The Open Power System Data time series package (version 2020-10-06, DOI 10.25832/time_series/2020-10-06) and the
When2Heat package (version 2023-07-27, DOI 10.25832/when2heat/2023-07-27) are supported by `load_tier_a` in
`src/vita/data.py`. The host `data.open-power-system-data.org` was unreachable from the computing environment, so
this path was not exercised.

## Techno-economic data

PyPSA technology-data, release v0.13.0, projection year 2030 (Zenodo DOI 10.5281/zenodo.15640601), file
`outputs/costs_2030.csv` (265,861 bytes, SHA-256
`0407b2c60ee2cabb06064648956f3b77ad1a8ab0e99d30aeb5eafa09772c5573`), released under CC-BY-4.0.
