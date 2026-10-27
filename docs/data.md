# Data provenance

Provider: NASA Ames Prognostics Center of Excellence. Citation: A. Saxena and K. Goebel (2008), “Turbofan Engine Degradation Simulation Data Set,” NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Paper: A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” First International Conference on Prognostics and Health Management, 2008.

The [NASA PCoE page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) links to this [PHM-hosted archive](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip). It was fetched on 27 October 2026. This degradation dataset is distinct from the separately listed unavailable C-MAPSS simulator software.

| File | SHA-256 |
| --- | --- |
| Outer archive | `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` |
| Inner `CMAPSSData.zip` | `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f` |
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

The downloader verifies archive/member hashes, reads only named FD001 members, and writes fixed filenames. It never extracts arbitrary archive paths. Download size is bounded at 30 MB; extraction starts only after the checksum passes. Different local data are not replaced. Changed upstream bytes fail closed and need a reviewed provenance update.

Raw data are excluded from Git. Custom correctly formatted files can be used, but their report says `custom-unverified`; they are not a NASA benchmark. The official files have 100 training engines/20,631 rows and 100 test engines/13,096 rows, plus 100 endpoint RUL offsets.

Each row has unit ID, cycle, three settings, and 21 sensors. Readers require 26 finite columns, positive integer IDs/cycles, contiguous unit IDs, and complete cycle sequences starting at 1. Offsets must be one nonnegative integer per test engine. These are simulation outputs, not measurements collected by this project.

NASA's [metadata](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) does not specify a dataset license. The MIT code license does not cover NASA data. Data are downloaded separately and credited. Check the requirements for your intended use and redistribution. No NASA endorsement is claimed.
