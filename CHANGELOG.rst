DIALS 3.1.3 (2020-09-28)
========================

Bugfixes
--------

- ``dials.integrate``: fix integrator=3d_threaded crash if njobs > 1 (`#1410 <https://github.com/dials/dials/issues/1410>`_)
- ``dials.integrate``: Check for and show error message if shoebox data is missing (`#1421 <https://github.com/dials/dials/issues/1421>`_)
- ``dials.refine``: Avoid crash for experiments with zero reflections if the
  `auto_reduction.action=remove` option was active (`#1417 <https://github.com/dials/dials/issues/1417>`_)

Improved Documentation
----------------------

- ``dials.merge``: improve help message by adding usage examples (`#1413 <https://github.com/dials/dials/issues/1413>`_)
- ``dials.refine``: More helpful error message when too few reflections (`#1431 <https://github.com/dials/dials/issues/1431>`_)


DIALS 3.1.2 (2020-09-14)
========================

Features
--------

- ``dials.stills_process``: Add parameter ``max_images=`` to limit the number
  of processed images

Bugfixes
--------

- ``dials.integrate``: fix crash when run with integrator=3d_threaded (`#1404 <https://github.com/dials/dials/issues/1404>`_)
- ``dials.integrate``: Minor performance improvements (`#1399 <https://github.com/dials/dials/issues/1399>`_)
- ``dials.stills_process``: MPI performance improvements for large datasets
- ``dials.stills_process``: Fix error when using split logs


DIALS 3.1.1 (2020-09-01)
========================

Bugfixes
--------

- ``dials.scale``: Prevent discarding of resolution limits in rare cases, which
  could cause incorrect symmetry determination, and worse results. (`#1378 <https://github.com/dials/dials/issues/1378>`_)
- ``dials.cosym``: filter out experiments with inconsistent unit cells (`#1380 <https://github.com/dials/dials/issues/1380>`_)
- Internally slicing experiments now works if image range doesn't start at 1 (`#1383 <https://github.com/dials/dials/issues/1383>`_)
- Restore missing I/sigma(I) resolution estimate log output (`#1384 <https://github.com/dials/dials/issues/1384>`_)
- ``dials.image_viewer``: "Save As" button now works, for single panels
- Fix developer ``libtbx.precommit`` installation error (`#1375 <https://github.com/dials/dials/issues/1375>`_)


DIALS 3.1 (2020-08-17)
======================

Features
--------

- Supports Python 3.7 and 3.8. Python 3.6 remains the default. (`#1236 <https://github.com/dials/dials/issues/1236>`_)
- Switch DIALS environment to use conda compilers. For development environments,
  a new ``dials`` script, located above the build directory, replaces the
  existing 'setpaths'-family of scripts. This means that all commands within
  the conda environment will now be available. (`#1235 <https://github.com/dials/dials/issues/1235>`_)
- New command: ``dials.missing_reflections`` to identify connected regions of
  missing reflections in the asymmetric unit. (`#1285 <https://github.com/dials/dials/issues/1285>`_)
- Improvements to image stacking in ``dials.image_viewer``:
  - add pull-down selector for stacking mode
  - add modes for mean and max
  - add command-line selection for stacking mode
  - rename ``sum_images`` command-line option to ``stack_images`` (`#1302 <https://github.com/dials/dials/issues/1302>`_)
- Reduce volume of output in ``dials.integrate``; histograms and other less
  important information only shows in debug output. Pass the ``-vv`` option
  to restore the previous behaviour (`#1319 <https://github.com/dials/dials/issues/1319>`_)
- ``dials.integrate``: Experimental feature: Specifying
  ``output_unintegrated_reflections=False`` discards unintegrated data from
  output reflection file, for smaller output and faster post-processing (`#1343 <https://github.com/dials/dials/issues/1343>`_)
- Rename ``dials.resolutionizer`` command to ``dials.estimate_resolution``,
  and includes a html report. Writing png plot output is now turned off by
  default (passing ``plot=True`` will restore this behaviour). (`#1330 <https://github.com/dials/dials/issues/1330>`_)
- ``dials.scale`` now separates anomalous pairs during error model analysis (`#1332 <https://github.com/dials/dials/issues/1332>`_)
- ``dials.background``: Add parameter ``corrected=`` to optionally use
  pedestal-and-gain corrected data (`#1348 <https://github.com/dials/dials/issues/1348>`_)
- ``dials.combine_experiments``: Add option ``output.max_reflections_per_experiment=``,
  to reject experiments with too many reflections (`#1369 <https://github.com/dials/dials/issues/1369>`_)


Bugfixes
--------

- ``dials.image_viewer``: Shoeboxes are now shown rotated with rotated detector panels. (`#1189 <https://github.com/dials/dials/issues/1189>`_)
- ``dials.index``: In multi-lattice indexing, ensure that reflections where
  refinement fails are flagged as unindexed. (`#1350 <https://github.com/dials/dials/issues/1350>`_)
- ``dials.scale``: Reflections excluded from scaling are no longer permanently
  excluded from any subsequent ``dials.scale`` jobs. (`#1275 <https://github.com/dials/dials/issues/1275>`_)
- ``dials.scale``: When using ``intensity_choice=combine`` (the default), don't
  exclude reflections that only have one of summed or profiled intensities
  available, but not both. (`#1300 <https://github.com/dials/dials/issues/1300>`_)
- ``dials.split_experiments``: Don't generate extra leading zeros in the output
  filename when not required e.g. ``output_09.expt`` -> ``output_9.expt`` (`#1316 <https://github.com/dials/dials/issues/1316>`_)
- ``dials.plot_reflections``: Fix invisible white spots on white background. (`#1346 <https://github.com/dials/dials/issues/1346>`_)


Deprecations and Removals
-------------------------

- ``dials.find_spots``: Deprecate ``spotfinder.filter.use_trusted_range=`` (`#1156 <https://github.com/dials/dials/issues/1156>`_)
- ``setpaths.sh`` and related scripts in newly created DIALS development
  environments are made obsolete and will no longer work. (`#1235 <https://github.com/dials/dials/issues/1235>`_)
- ``dials.show``: Remove ``show_image_statistics=`` parameter. Use
  ``image_statistics.show_raw=`` for equivalent output (`#1306 <https://github.com/dials/dials/issues/1306>`_)
- Log files will omit timings unless the relevant dials program was run with ``-v`` (`#1313 <https://github.com/dials/dials/issues/1313>`_)

Misc
----

- `#1184 <https://github.com/dials/dials/issues/1184>`_, `#1216 <https://github.com/dials/dials/issues/1216>`_, `#1288 <https://github.com/dials/dials/issues/1288>`_, `#1312 <https://github.com/dials/dials/issues/1312>`_, `#1320 <https://github.com/dials/dials/issues/1320>`_, `#1322 <https://github.com/dials/dials/issues/1322>`_, `#1325 <https://github.com/dials/dials/issues/1325>`_, `#1328 <https://github.com/dials/dials/issues/1328>`_, `#1352 <https://github.com/dials/dials/issues/1352>`_, `#1365 <https://github.com/dials/dials/issues/1365>`_, `#1366 <https://github.com/dials/dials/issues/1366>`_, `#1370 <https://github.com/dials/dials/issues/1370>`_


DIALS 3.0.4 (2020-07-20)
========================

- ``dials.scale``: Allow usage of ``mode=image_group`` with ``filtering.method=deltacchalf`` when
  only providing a single data set (`#1334 <https://github.com/dials/dials/issues/1334>`_)
- ``dials.import``: When using a template and specifying an image_range, missing images outside of
  the range will not cause a failure (`#1333 <https://github.com/dials/dials/issues/1333>`_)
- ``dials.stills_process``: Show better error message in specific spotfinding failure case (`#1180 <https://github.com/dials/dials/issues/1180>`_)


DIALS 3.0.3 (2020-07-06)
========================

Features
--------

- Developer tool: On posix systems, sending SIGUSR2 to DIALS commands will now print a stack trace (`#1277 <https://github.com/dials/dials/issues/1277>`_)

Bugfixes
--------
- HTML reports: Plot bin centres instead bin minimum for d_min line plots vs. resolution (`#1323 <https://github.com/dials/dials/issues/1323>`_)
- ``dials.export``: Fix inconsistency in mtz export when given a non-reference (e.g. I2 or primitive) setting (`#1279 <https://github.com/dials/dials/issues/1279>`_)
- ``dials.refine_bravais_settings``: Fix crash with large (>2gb) reflection tables and reduce memory use (`#1274 <https://github.com/dials/dials/issues/1274>`_)
- ``dials.scale``: Fix bug in outlier rejection code causing misidentification of outliers (with outlier_rejection=standard).
- ``dials.scale``: Fix outlier rejection formula to avoid overconfidence in spuriously low values


DIALS 3.0.2 (2020-06-23)
========================

Bugfixes
--------

- Fix crash in scaling error model handling (`#1243 <https://github.com/dials/dials/issues/1243>`_)


DIALS 3.0.1 (2020-06-11)
========================

Features
--------

- dials.reciprocal_lattice_viewer: Add an option to show lattice(s) in the crystal rather than laboratory frame. (`#1259 <https://github.com/dials/dials/issues/1259>`_)
- Support for mtz project_name in export and scaling

Bugfixes
--------

- dials.reciprocal_lattice_viewer: fix multiple experiment view for integrated data (`#1284 <https://github.com/dials/dials/issues/1284>`_)


DIALS 3.0 (2020-05-22)
======================

Features
--------

- Show more useful output when crashing in C++ code (`#659 <https://github.com/dials/dials/issues/659>`_)
- dials.image_viewer: for the unit cell tool, rename parameters for consistency and add a new show_hkl option to filter displayed powder rings to select only those of interest. (`#1192 <https://github.com/dials/dials/issues/1192>`_)
- In dials.integrate: changed the background box size multiplier to be a parameter (sigma_b_multiplier) - setting to small values significantly reduces memory requirements. (`#1195 <https://github.com/dials/dials/issues/1195>`_)
- dials.image_viewer: add an overlaying showing pixels marked as strong by the spot-finding operations. That is, the pixels picked out by the "threshold" image. (`#1200 <https://github.com/dials/dials/issues/1200>`_)
- dials.scale report file was renamed from scaling.html to dials.scale.html
  dials.symmetry report file was renamed from dials-symmetry.html to dials.symmetry.html (`#1202 <https://github.com/dials/dials/issues/1202>`_)
- dials.report output file was renamed from dials-report.html to dials.report.html (`#1206 <https://github.com/dials/dials/issues/1206>`_)
- dials.image_viewer: faster navigation between different image types. (`#1213 <https://github.com/dials/dials/issues/1213>`_)
- Crystal model now has a new recalculated_unit_cell attribute. This allows it to store
  a post-refined unit cell (e.g. from dials.two_theta_refine) in addition to that from
  traditional geometry refinement (which was used for prediction). Downstream programs
  such as dials.scale and dials.export will now use the recalculated unit cell 
  where appropriate. (`#1214 <https://github.com/dials/dials/issues/1214>`_)
- New best_monoclinic_beta parameter for dials.refine_bravais_settings and dials.symmetry.
  Setting this to False will ensure that C2 is selected in preference to I2, where I2
  would lead to a less oblique cell (i.e. smaller beta angle). (`#1226 <https://github.com/dials/dials/issues/1226>`_)
- New scaling model, model=dose_decay, implementing a shared exponential decay component for multicrystal experiments (`#1183 <https://github.com/dials/dials/issues/1183>`_)


Bugfixes
--------

- Avoid empty "Unable to handle" messages on failed dials.import (`#600 <https://github.com/dials/dials/issues/600>`_)
- Functions from dials.export now raise exceptions on errors rather than exit. This improves their use elsewhere (such as in dials.scale). (`#1205 <https://github.com/dials/dials/issues/1205>`_)
- Ensure dials.index chooses the C2 setting with the smallest beta angle (`#1217 <https://github.com/dials/dials/issues/1217>`_)
- Fix propagation of best_unit_cell and application of resolution cutoffs in dials.scale and export_mtz.
  Add a new mtz.best_unit_cell parameter to dials.export (`#1248 <https://github.com/dials/dials/issues/1248>`_)
- Make some of the DIALS tools furthest downstream (``dials.scale``, ``dials.symmetry``, ``dials.merge`` and ``dials.report``) more robust in the case of very few reflections. (`#1263 <https://github.com/dials/dials/issues/1263>`_)


Misc
----

- `#1221 <https://github.com/dials/dials/issues/1221>`_


DIALS 2.2 (2020-03-15)
======================

Features
--------

- dials.image_viewer: Add a choice between displaying the raw or the corrected image. (`#634 <https://github.com/dials/dials/issues/634>`_)
- Automatically choose between the `simple` and `glm` background determination
  algorithms, depending on whether the detector appears to be integrating or
  counting. (`#706 <https://github.com/dials/dials/issues/706>`_)
- Allow adjustment of font size for overlay text, such as Miller indices and
  resolution ring values. (`#1074 <https://github.com/dials/dials/issues/1074>`_)
- Keep goniometer and scan objects in indexing of still data, if provided in input (`#1076 <https://github.com/dials/dials/issues/1076>`_)
- Experimental: ``dials.image_viewer`` can be remotely controlled via a
  ZeroMQ endpoint with the ``zmq_endpoint`` PHIL parameter. Initially,
  the viewer can be commanded to load new images. This requires the
  (optional) ``pyzmq``package. (`#1085 <https://github.com/dials/dials/issues/1085>`_)
- Programs now generate a unique identifier for each experiment created, and reflection tables are linked via the experiment_identifiers map (`#1086 <https://github.com/dials/dials/issues/1086>`_)
- Introduce `dials.anvil_correction` to correct the absorption of the incident and diffracted X-ray beam by the diamond anvils in a pressure cell.
  Call `dials.anvil_correction` on the output of `dials.integrate` and then proceed to use post-integration tools as normal, just as though the sample had been measured in air. (`#1090 <https://github.com/dials/dials/issues/1090>`_)
- Map of detector efficiency for photon counting detectors as a function of 
  detector position added to report, based on the qe value applied at the end 
  of integration. (`#1108 <https://github.com/dials/dials/issues/1108>`_)
- Significantly reduce the amount of memory required to write .refl output files (`#1115 <https://github.com/dials/dials/issues/1115>`_)
- Add maximum_trusted_value=N option to spot finding to temporarily allow override of trusted range, e.g. to find overloaded spots in spot finding. (`#1157 <https://github.com/dials/dials/issues/1157>`_)
- array_family.flex interface has changed: background and centroid algorithms are
  set via public properties. Instead of flex.strategy use functools.partial with
  the same signature. as_miller_array() raises KeyError instead of Sorry.
  .extract_shoeboxes() lost its verbosity parameter, use log levels instead. (`#1158 <https://github.com/dials/dials/issues/1158>`_)
- dials.stills_process now supports imagesets of length > 1 (e.g. grid scans) (`#1174 <https://github.com/dials/dials/issues/1174>`_)


Bugfixes
--------

- Fixed prediction on images numbered zero, so integrating works correctly. (`#1128 <https://github.com/dials/dials/issues/1128>`_)
- Fix an issue (`#1097 <https://github.com/dials/dials/issues/1097>`_) whereby aggregating small numbers of reflections into resolution bins could sometimes result in empty bins and consequent errors. (`#1130 <https://github.com/dials/dials/issues/1130>`_)
- Ensure that restraints are ignored for parameterisations that are anyway fixed (`#1142 <https://github.com/dials/dials/issues/1142>`_)
- Fix dials.search_beam_centre to ensure that the correct detector models are
  output when multiple detector models are present in the input.
  Fix dials.search_beam_centre n_macro_cycles option (previously it was starting
  from the original geometry every macro cycle). (`#1145 <https://github.com/dials/dials/issues/1145>`_)
- dials.find_spots_server no longer slows down 3x when using resolution filters (`#1170 <https://github.com/dials/dials/issues/1170>`_)


Misc
----

- `#932 <https://github.com/dials/dials/issues/932>`_, `#1034 <https://github.com/dials/dials/issues/1034>`_, `#1050 <https://github.com/dials/dials/issues/1050>`_, `#1077 <https://github.com/dials/dials/issues/1077>`_


DIALS 2.1 (2019-12-12)
======================

Features
--------

- We now fully support Python 3 environments.
- MessagePack is now the default reflection table file format. Temporarily, the
  environment variable ``DIALS_USE_PICKLE`` can be used to revert to the previous
  pickle-based format, however this will be removed in a future version. (`#986 <https://github.com/dials/dials/issues/986>`_)
- new option for dials.show 'show_shared_models=True' displays which beam, crystal, and detector models are used across experiments (`#996 <https://github.com/dials/dials/issues/996>`_)
- Import still image sequence as N experiments dereferencing into one image set
  rather than one experiment. (`#1014 <https://github.com/dials/dials/issues/1014>`_)
- Add `reflection_table.get` method for defaulted column access (`#1031 <https://github.com/dials/dials/issues/1031>`_)


Bugfixes
--------

- Don't use -2 to indicate masked pixels, except for DECTRIS detectors where this
  is to be expected. (`#536 <https://github.com/dials/dials/issues/536>`_)
- No longer show pixels that are above the trusted range upper bound as
  "saturated" on the "variance" image. (`#846 <https://github.com/dials/dials/issues/846>`_)
- Correctly account for scan-varying crystals while providing a scan range to
  dials.integrate (`#962 <https://github.com/dials/dials/issues/962>`_)
- Ensure that generated masks do not include pixels that are overloaded on a few
  images, but only pixels that are always outside the trusted range. (`#978 <https://github.com/dials/dials/issues/978>`_)
- Rewritten parameter auto-reduction code for dials.refine provides finer-grained
  fixing of individual parameters rather than whole parameterisations and
  correctly takes constrained parameters into account (`#990 <https://github.com/dials/dials/issues/990>`_)
- Fix output of predictions in dials.refine.
  A recently-introduced bug meant that the updated predictions weren't
  being copied to the output reflections file. (`#991 <https://github.com/dials/dials/issues/991>`_)
- Allow scan-varying refinement where either the crystal cell or
  orientation is fixed. (`#999 <https://github.com/dials/dials/issues/999>`_)
- Respect batch= option to dials.symmetry - can reduce time taken for finding
  the symmetry for large data sets. (`#1000 <https://github.com/dials/dials/issues/1000>`_)
- Scan-varying refinement no longer fails when the scan is wider than the
  observed reflections (e.g. when the crystal has died). Instead, the scan
  is first trimmed to match the range of the diffraction. (`#1025 <https://github.com/dials/dials/issues/1025>`_)
- If convert_sequences_to_stills then delete the goniometer and scan. (`#1035 <https://github.com/dials/dials/issues/1035>`_)
- Correctly account for scan-varying crystals in dials.slice_sequence (`#1040 <https://github.com/dials/dials/issues/1040>`_)
- Eliminate systematic absences before applying change of basis op to minimum 
  cell in dials.symmetry. (`#1064 <https://github.com/dials/dials/issues/1064>`_)


Improved Documentation
----------------------

- Add "Extending DIALS" page to developer documentation (`#893 <https://github.com/dials/dials/issues/893>`_)


Deprecations and Removals
-------------------------

- The command dials.analyse_output was removed.
  Its replacement, dials.report, will give you more useful output. (`#1009 <https://github.com/dials/dials/issues/1009>`_)


Misc
----

- `#983 <https://github.com/dials/dials/issues/983>`_, `#1004 <https://github.com/dials/dials/issues/1004>`_


DIALS 2.0 (2019-10-23)
======================

Features
--------

- Support exporting multi-dataset and still experiments to XDS_ASCII (`#637 <https://github.com/dials/dials/issues/637>`_)
- Replace default spotfinder with improved dispersion algorithm (`#758 <https://github.com/dials/dials/issues/758>`_)
- ``dials.report`` now displays oscillation data with units and more significant figures (`#896 <https://github.com/dials/dials/issues/896>`_)
- A new program, ``dials.sequence_to_stills`` to create split a sequence into a
  separate still Experiment for every scan point in the sequence, splitting
  reflections as necessary. (`#917 <https://github.com/dials/dials/issues/917>`_)
- Moved ``dials.export format=best`` to ``dials.export_best`` as that one needed
  access to the format object, the rest do not, and having ``dials.export`` work
  in the general case seems like a better idea... (`#921 <https://github.com/dials/dials/issues/921>`_)
- Unified logging output for dials programs - logs are no longer split into .log
  and .debug.log. Use -v to get debug output. (`#923 <https://github.com/dials/dials/issues/923>`_)
- New command ``dials.resolutionizer`` (replaces ``xia2.resolutionizer``). Add support for ``expt``/``refl``
  in ``dials.resolutionizer``. (`#933 <https://github.com/dials/dials/issues/933>`_)
- Changed the selection of reflections used for determination of the reflection
  profile parameters in integration. Now uses reflections which were previously
  used in refinement rather than all reflections, resulting in a speed
  improvement for large data sets and a negligible difference in the quality
  of the integrated results. (`#942 <https://github.com/dials/dials/issues/942>`_)
- ``dials.image_viewer`` now allows the choice between
  ``dispersion_extended`` (new default) and ``dispersion`` (old default)
  thresholding algorithms for investigating the effect of different
  spot-finding parameters. (`#948 <https://github.com/dials/dials/issues/948>`_)
- ``dials.rs_mapper`` now respects masked regions of images (including
  the trusted range mask). (`#955 <https://github.com/dials/dials/issues/955>`_)


Bugfixes
--------

- Fix and reinstate normalisation option in ``dials.option`` (`#919 <https://github.com/dials/dials/issues/919>`_)


Misc
----

- `#795 <https://github.com/dials/dials/issues/795>`_, `#862 <https://github.com/dials/dials/issues/862>`_, `#895 <https://github.com/dials/dials/issues/895>`_, `#915 <https://github.com/dials/dials/issues/915>`_, `#924 <https://github.com/dials/dials/issues/924>`_
