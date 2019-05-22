#!/usr/bin/env python
#
# dials.potato.py
#
#  Copyright (C) 2018 Diamond Light Source
#
#  Author: James Parkhurst
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

# LIBTBX_SET_DISPATCHER_NAME dials.potato

from __future__ import absolute_import, division
import libtbx.load_env
from libtbx.utils import Sorry
from libtbx.phil import parse
from dials.util import log
from dials.util.options import OptionParser
from dials.util.options import flatten_reflections, flatten_experiments
from dials_scratch.jmp.potato.potato import Integrator
from dxtbx.model.experiment_list import ExperimentListDumper
import logging


logger = logging.getLogger(libtbx.env.dispatcher_name)

help_message = """

This script does profile modelling for stills

"""

# Create the phil scope
phil_scope = parse(
    """

  output {
    experiments = "integrated_experiments.json"
      .type = str
      .help = "The output experiments"

    reflections = "integrated.pickle"
      .type = str
      .help = "The output reflections"
  }

  include scope dials_scratch.jmp.potato.potato.phil_scope

""",
    process_includes=True,
)


class Script(object):
    """
  The integration program.

  """

    def __init__(self):
        """
    Initialise the script.

    """

        # The script usage
        usage = "usage: %s [options] experiment.json" % libtbx.env.dispatcher_name

        # Create the parser
        self.parser = OptionParser(
            usage=usage,
            phil=phil_scope,
            epilog=help_message,
            read_experiments=True,
            read_reflections=True,
        )

    def run(self):
        """
    Perform the integration.

    """
        from time import time

        # Check the number of arguments is correct
        start_time = time()

        # Parse the command line
        params, options = self.parser.parse_args(show_diff_phil=False)
        reflections = flatten_reflections(params.input.reflections)
        experiments = flatten_experiments(params.input.experiments)
        if len(reflections) == 0 or len(experiments) == 0:
            self.parser.print_help()
            return
        elif len(reflections) != 1:
            raise Sorry("more than 1 reflection file was given")
        elif len(experiments) == 0:
            raise Sorry("no experiment list was specified")
        reflections = reflections[0]

        # Configure logging
        log.config(info="dials.potato.log", debug="dials.potato.debug.log")

        # Log the diff phil
        diff_phil = self.parser.diff_phil.as_str()
        if diff_phil is not "":
            logger.info("The following parameters have been modified:\n")
            logger.info(diff_phil)

        # Contruct the integrator
        integrator = Integrator(experiments, reflections, params)

        # Do cycles of indexing and refinement
        for i in range(params.refinement.n_macro_cycles):
            integrator.reindex_strong_spots()
            integrator.integrate_strong_spots()
            integrator.refine()

        # Do the integration
        integrator.predict()
        integrator.integrate()

        # Get the reflections
        reflections = integrator.reflections
        experiments = integrator.experiments

        # Save the reflections
        reflections.as_pickle(params.output.reflections)

        dump = ExperimentListDumper(experiments)
        with open(params.output.experiments, "w") as outfile:
            outfile.write(dump.as_json())


if __name__ == "__main__":
    from dials.util import halraiser

    try:
        script = Script()
        script.run()
    except Exception as e:
        halraiser(e)
