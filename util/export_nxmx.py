from __future__ import division

def export_beam(outfile, beam):
  ''' Export the beam model. '''
  from scitbx import matrix

  EPS = 1e-7

  # Make sure that the direction is 0, 0, -1
  assert(matrix.col(beam.get_direction()).dot(matrix.col((0, 0, -1))) < EPS)

  # Get the nx_beam
  nx_beam = outfile.entry.sample.beam
  nx_beam['incident_wavelength'] = beam.get_wavelength()


def export_detector(outfile, detector):
  ''' Export the detector model. '''
  from scitbx import matrix

  # Get the panel
  panel = detector[0]

  # Get some panel attributes
  pixel_size = panel.get_pixel_size()
  image_size = panel.get_image_size()
  origin = matrix.col(panel.get_origin())

  # Get the detector module object
  detector = outfile.entry.instrument.detector
  module = outfile.entry.instrument.detector.module['module0']

  # Set the data size
  module['data_size'] = image_size

  # Create the detector translation
  transformations = detector.transformations
  transformations['translation'] = 0
  transformations['translation'].attrs['depends_on'] = '.'
  transformations['translation'].attrs['transformation_type'] = 'translation'
  transformations['translation'].attrs['units'] = 'mm'
  transformations['translation'].attrs['vector'] = (0, 0, 1)

  # Get the path for below
  translation_path = str('%s/%s' % (transformations.path(), 'translation'))

  # Create the detector depends on
  detector['depends_on'] = translation_path

  # Set the module offset
  module['module_offset'] = origin.length()
  module['module_offset'].attrs['depends_on'] = translation_path
  module['module_offset'].attrs['transformation_type'] = 'translation'
  module['module_offset'].attrs['units'] = 'mm'
  module['module_offset'].attrs['vector'] = origin.normalize()

  # The path for items below
  module_offset_path = str('%s/%s' % (module.path(), 'module_offset'))

  # Write the fast pixel direction
  module['fast_pixel_direction'] = pixel_size[0]
  module['fast_pixel_direction'].attrs['depends_on'] = module_offset_path
  module['fast_pixel_direction'].attrs['transformation_type'] = 'translation'
  module['fast_pixel_direction'].attrs['units'] = 'mm'
  module['fast_pixel_direction'].attrs['vector'] = panel.get_fast_axis()

  # Write the slow pixel direction
  module['slow_pixel_direction'] = pixel_size[1]
  module['slow_pixel_direction'].attrs['depends_on'] = module_offset_path
  module['slow_pixel_direction'].attrs['transformation_type'] = 'translation'
  module['slow_pixel_direction'].attrs['units'] = 'mm'
  module['slow_pixel_direction'].attrs['vector'] = panel.get_slow_axis()

  # Write the fast pixel size
  module['fast_pixel_size'] = pixel_size[0]
  module['fast_pixel_size'].attrs['depends_on'] = module_offset_path
  module['fast_pixel_size'].attrs['transformation_type'] = 'translation'
  module['fast_pixel_size'].attrs['units'] = 'mm'
  module['fast_pixel_size'].attrs['vector'] = panel.get_fast_axis()

  # Write the slow pixel size
  module['slow_pixel_size'] = pixel_size[1]
  module['slow_pixel_size'].attrs['depends_on'] = module_offset_path
  module['slow_pixel_size'].attrs['transformation_type'] = 'translation'
  module['slow_pixel_size'].attrs['units'] = 'mm'
  module['slow_pixel_size'].attrs['vector'] = panel.get_slow_axis()

def export_goniometer(outfile, goniometer, scan):
  ''' Export the goniometer model. '''

  # The angles for each image
  phi0, dphi = scan.get_oscillation(deg=True)
  phi = [phi0+dphi*i for i in range(len(scan))]

  # Write out the rotation axis and oscillation
  transformations = outfile.entry.sample.transformations
  transformations['phi'] = phi
  transformations['phi'].attrs['depends_on'] = '.'
  transformations['phi'].attrs['transformation_type'] = 'rotation'
  transformations['phi'].attrs['units'] = 'deg'
  transformations['phi'].attrs['vector'] = goniometer.get_rotation_axis()

def export_crystal(outfile, crystal):
  ''' Export the crystal model. '''

  from scitbx.array_family import flex

  # Get the sample
  sample = outfile.entry.sample

  # Set the space group
  sample['unit_cell_group'] = crystal.get_space_group().type().hall_symbol()

  # Get the unit cell and orientation matrix in the case of scan varying and
  # scan static models
  if crystal.num_scan_points:
    num = crystal.num_scan_points
    unit_cell = flex.double(flex.grid(num, 6))
    orientation_matrix = flex.double(flex.grid(num, 9))
    for i in range(num):
      __cell = crystal.get_unit_cell_at_scan_point(i).parameters()
      for j in range(6):
        unit_cell[i,j] = __cell[j]
      __matrix = crystal.get_U_at_scan_point(i).transpose().elems
      for j in range(9):
        orientation_matrix[i,j] = __matrix[j]

  else:
    unit_cell = crystal.get_unit_cell().parameters()
    orientation_matrix = crystal.get_U().transpose()

  # Save the unit cell data
  sample['unit_cell'] = unit_cell
  sample['unit_cell'].attrs['angles_units'] = 'deg'
  sample['unit_cell'].attrs['length_units'] = 'angstrom'

  # Save the orientation matrix
  sample['orientation_matrix'] = orientation_matrix

  # Set depends on
  sample['depends_on'] = str('%s/%s' % (sample.transformations.path(), 'phi'))


def export_experiments(outfile, experiments):
  ''' Export the experiments to the NXmx file. '''

  # Ensure only 1 experiment at the moment
  assert(len(experiments) == 1)
  experiment = experiments[0]

  # Ensure only 1 panel at the moment
  assert(len(experiment.detector) == 1)

  # Do crystal change of basis
  space_group_info = experiment.crystal.get_space_group().info()
  cb_op_to_ref = space_group_info.change_of_basis_op_to_reference_setting()
  crystal = experiment.crystal.change_basis(cb_op_to_ref)

  # Export the beam model
  export_beam(outfile, experiment.beam)

  # Export the detector
  export_detector(outfile, experiment.detector)

  # Export the goniometer
  export_goniometer(outfile, experiment.goniometer, experiment.scan)

  # Export the crystal
  export_crystal(outfile, crystal)

def export_reflections(outfile, reflections):
  ''' Export the reflection table to the NXmx file. '''

  # Ensure only a single experiment
  assert(reflections['id'].all_eq(0))

  # Ensure only a single panel
  assert(reflections['panel'].all_eq(0))

  # Select only those reflections which have valid summed intensities and where
  # present, profile fitted intensities. Also only export fully recorded
  # reflections
  selection = reflections['intensity.sum.variance'] <= 0
  if selection.count(True) > 0:
    reflections.del_selected(selection)
    print 'Removing %d reflections with negative variance...' % \
          selection.count(True)

  if 'intensity.prf.variance' in reflections:
    selection = reflections['intensity.prf.variance'] <= 0
    if selection.count(True) > 0:
      reflections.del_selected(selection)
      print 'Removing %d profile reflections with negative variance...' % \
            selection.count(True)

  if 'partiality' in reflections:
    selection = reflections['partiality'] < 0.99
    if selection.count(True) > 0:
      reflections.del_selected(selection)
      print 'Removing %d incomplete reflections...' % \
        selection.count(True)

  # Get the diffraction class
  diffraction = outfile.entry.diffraction

  # Export all the columns
  try:
    col1, col2, col3 = zip(*list(reflections['miller_index']))
    diffraction['h'] = col1
    diffraction['k'] = col2
    diffraction['l'] = col3
  except Exception:
    pass

  try:
    diffraction['id'] = reflections['id']
  except Exception:
    pass

  try:
    diffraction['int_sum_val'] = reflections['intensity.sum.value']
    diffraction['int_sum_var'] = reflections['intensity.sum.variance']
  except Exception:
    pass

  try:
    diffraction['int_prf_val'] = reflections['intensity.prf.value']
    diffraction['int_prf_var'] = reflections['intensity.prf.variance']
  except Exception:
    pass

  try:
    diffraction['lp'] = reflections['lp']
  except Exception:
    pass

  try:
    diffraction['det_module'] = reflections['panel']
  except Exception:
    pass

  try:
    col11, col12, col13, col14, col15, col16 = reflections['bbox'].parts()
    diffraction['bbx0'] = col11
    diffraction['bbx1'] = col12
    diffraction['bby0'] = col13
    diffraction['bby1'] = col14
    diffraction['bbz0'] = col15
    diffraction['bbz1'] = col16
  except Exception:
    pass

  try:
    col17, col18, col19 = reflections['xyzcal.px'].parts()
    diffraction['prd_px_x'] = col17
    diffraction['prd_px_y'] = col18
    diffraction['prd_frame'] = col19
  except Exception:
    pass

  try:
    col20, col21, col22 = reflections['xyzcal.mm'].parts()
    diffraction['prd_mm_x'] = col20
    diffraction['prd_mm_y'] = col21
    diffraction['prd_phi'] = col22
  except Exception:
    pass

  try:
    col23, col24, col25 = reflections['xyzobs.px.value'].parts()
    col26, col27, col28 = reflections['xyzobs.px.variance'].parts()
    diffraction['obs_px_x_val'] = col23
    diffraction['obs_px_x_var'] = col24
    diffraction['obs_px_y_val'] = col25
    diffraction['obs_px_y_var'] = col26
    diffraction['obs_frame_val'] = col27
    diffraction['obs_frame_var'] = col28
  except Exception:
    pass

  try:
    col29, col30, col31 = reflections['xyzobs.mm.value'].parts()
    col32, col33, col34 = reflections['xyzobs.mm.variance'].parts()
    diffraction['obs_mm_x_val'] = col29
    diffraction['obs_mm_x_var'] = col30
    diffraction['obs_mm_y_val'] = col31
    diffraction['obs_mm_y_var'] = col32
    diffraction['obs_phi_val'] = col33
    diffraction['obs_phi_var'] = col34
  except Exception:
    pass

  try:
    diffraction['partiality'] = reflections['partiality']
  except Exception:
    pass

  try:
    diffraction['d'] = reflections['d']
  except Exception:
    pass

  try:
    diffraction['bkg_mean'] = reflections['background.mean']
  except Exception:
    pass

  try:
    diffraction['entering'] = reflections['entering']
  except Exception:
    pass

  try:
    diffraction['flags'] = reflections['flags']
  except Exception:
    pass

  try:
    diffraction['prf_cc'] = reflections['profile.correlation']
  except Exception:
    pass

def export_details(outfile):
  from time import strftime

  # Get the entry
  entry = outfile.entry

  # Program info
  entry['program_name'] = 'dials.export_nxmx'
  entry['program_name'].attrs['version'] = 1
  entry['program_name'].attrs['configuration'] = ''

  # Set some processing information (each program should add itself)
  process = entry.process['spotfinding']
  process['program'] = 'dials.find_spots'
  process['version'] = 1
  process['date'] = strftime('%Y-%m-%dT%H:%M:%S')

  process = entry.process['indexing']
  process['program'] = 'dials.index'
  process['version'] = 1
  process['date'] = strftime('%Y-%m-%dT%H:%M:%S')

  process = entry.process['refinement']
  process['program'] = 'dials.refine'
  process['version'] = 1
  process['date'] = strftime('%Y-%m-%dT%H:%M:%S')

  process = entry.process['integration']
  process['program'] = 'dials.integrate'
  process['version'] = 1
  process['date'] = strftime('%Y-%m-%dT%H:%M:%S')

def export(experiments, reflections, filename):
  ''' Export the experiments and reflections as an NXmx file. '''
  from dials.scratch.jmp.mtz2 import mtz2

  # Open the NXmx file for writing
  outfile = mtz2.File(filename, 'w')

  # Export the experiments
  print 'Exporting experimental models...'
  export_experiments(outfile, experiments)

  # Export the reflection table data
  print 'Exporting reflection data...'
  export_reflections(outfile, reflections)

  # Export some extra details
  print 'Exporting some additional processing details...'
  export_details(outfile)

  # Flush the file
  outfile.flush()
  print 'Wrote NXmx file %s' % filename

  # FIXME The following items have not been set from export_mtz
  # o.set_divhd(0.0).set_divvd(0.0)
  # o.set_bbfac(0.0).set_bscale(1.0)
  # o.set_sdbfac(0.0).set_sdbscale(0.0).set_nbscal(0)
  # o.set_lcrflg(0)
  # o.set_datum(flex.float((0.0, 0.0, 0.0)))
  # o.set_misflg(0)
  # o.set_jumpax(0)
  # o.set_ldtype(2)
