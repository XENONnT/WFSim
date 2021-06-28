v0.5.4 / 2021-06-24
===================
* Get gains from CMT (#172)
* Not to remove nVeto events whose related TPC events don't exist (#171)
* Remove travis for testing in wfsim (#176)
* Oversize chunk handling (#168)
* Adding Kr83m notebook (#167)
* Sorting of S1 photons by channel number to preserve timing (#169)

v0.5.3 / 2021-06-03
===================
* Truth chunk patch (#164)

v0.5.2 / 2021-05-25
===================
* Fix zstd for testing (#163)
* Patch 1T custom context (#162)
* Fixing SPE reconstruciton bias due to truncation (#161)
* Following up the change in pmt ap file (#158)
* Fix #157 for empty files (#160)
* Fix missing last instruction (#159)
* Fix load resource (#157)
* S2 photon patch (#156)
* Fix notebook, evaluate for docs (#154)
* Add tests (#152)
* Use database access to increase coverage of tests (#155)
* Update Getting_started_wfsim.ipynb (#153)

v0.5.1 / 2021-05-03
===================
* S1 timing splines implementation for top/bottom arrays (#148)
* Update classifiers (#150)
* Tuck load ele ap in tpc conditions (#149)
* Fix nveto readout (#147)
* Map update followup (#143)
* Add tests for #144 (#146)
* No default for gain_model_nv defined in the class (#145)

v0.5.0 / 2021-04-22
===================
* McChainSimulator (major) patch (#125)
* Update utilix version in requirements.txt (ffc6c2f)
* Do not track the timeout (#131)
* Change dtype of truth (#138)
* Load Rz field maps (#130)
* Conserve total quanta in tutorial (#135)
* Load GARFIELD map according to liquid level (#127)
* Fix PMT Afterpulse (#117)
* Fixing chunking issue (#118)
* Update history of releases (#123)
* Allow errors while building docs (#122)
* Fix missing epix import (#115)
* Fix non-literal compare (#121)
* Reintroduce self._electron_timings for truth (#116)
* Update tutorial notebook (f59c2c0 + ee6b6ad)

v0.4.1 / 2021-03-24
===================
* Miscellaneous fixes (#113)

v0.4.0 / 2021-03-16
===================
* S1+S2 functions are externally callable (#103)
* Docstrings for most functions (#103)
* Nveto QE implementation (#99)
* Functionality for full chain simulation (#111)
* Using epix for instruction clustering (#111)
* Website for documentation (#105)

v0.2.5 / 2021-02-26
===================
* Config patch and debugging prints (#104)

v0.2.4 / 2021-02-22
===================
* First pip installable release

v0.0.1
===================
* Release as a python package
