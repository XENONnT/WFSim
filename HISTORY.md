v1.0.0 / 2022-06-09
===================
* Start per pmt truth info (#369)
* S1 pulse shape debug (#381)
* Save Geant4 primary position in truth (#379)
* XY-dependent SE gain and extraction efficiency (#363)
* Fix broken tests (#375)
* Added scaling of S2 AFT according to value from config (#370)
* Off by one in noise (#364)
* Update requirements-tests.txt (#368)
* Fix zenodo badge 

v0.6.1 / 2022-04-12
===================
* Change default compressor (#361)
* Drift velocity maps using true interaction point (#360)

v0.6.0 / 2022-04-05
===================
* No dependabot in WFSim (#358)
* AP probability for photons emitting double PE (#340)
* FDC quick fix (#285)
* SE shape: optical propagation + anode focusing (#320)
* Fix CIV and average drift velocity implementation (#338)
* Test with straxen 1.5.0 (#337)
* Sort afterpulses by channel (#321)
* Reduce warnings (#317)
* Test with latest straxen (#322)
* Up py version (#316)

v0.5.13 / 2022-01-17
====================
* Save full truth output (#288)
* Bump strax(en) (#303)
* Use straxen#833 (#286)
* Integrated testing and requirements (#304, #289, #290, #291, #293, #294 #295, #296, #297)

v0.5.12 / 2021-12-10
====================
* Rename s2_aft_sigma config (#284)
* SPE spec bug fix, adding trigger truth (#283)
* Pin pymongo (#282)

v0.5.11 / 2021-11-24
====================
* Recompute lce on the run (#269)

v0.5.10 / 2021-11-19
====================
* Add n_pe to truth (#246)
* Fix nVeto dtypes handling in McChain (#262)
* Add infer_type=False to all strax.Options (#265)
* Bump straxen (#267)
* Skewed Gaussian for S2 AFT (#260) 
* Add zenodo badge (#259) 
* Field distortion and transverse diffusion map (#235)
* Remove deprecated config options (#247) 
* Run with latest strax(en) (#245)

v0.5.9 / 2021-10-29
===================
* Load MC pattern map if no data-driven LCE map provided (#241)
* Change LCE map for tests (#243)
* Typo in n_chunks (#238)
* Security fix for dask (#240)
* Remove not used nv option (#237)
* Pulse instrcutions notebook (#236)

v0.5.8 / 2021-10-19
===================
* Change to online cmt version (#234)
* Update rand instructions (#233)
* Add a link to nestpy for interaction types (#231)
* Enable data-driven maps and minor changes (#230)
* Make less ele ap instructions (#216)

v0.5.7 / 2021-08-27
===================
* Reinstate optical simulation (#199)
* Run with new strax (#198)
* Pinned tests (#197)
* S1 photon timing patching (#196)
* Bug fix for wrong place to write dtype (#195)
* Fix optical readout (#194)
* Only nveto mc chain (#190)
* Use real noise (#189)

v0.5.6 / 2021-07-19
===================
* Core splitting 2 (#188)
* Github McChain test using G4 file (#177)
* Fix removed nveto (#185)
* No context testing for py3.6 (#187)
* Overwrite detector parameters with CMT (#179)
* Fix the 0.5% bias in gain sampling (#186)

v0.5.5 / 2021-07-07
===================
* Bug fix in strax interface (#184)

v0.5.4 / 2021-06-24
===================
* Get gains from CMT (#172)
* Not to remove nVeto events whose related TPC events do not exist (#171)
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
