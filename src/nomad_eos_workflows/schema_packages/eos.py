from typing import TYPE_CHECKING, Optional
import numpy as np
import pint
import plotly.express as px

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.datamodel.data import Schema, ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.metainfo import MEnum, Quantity, SectionProxy, SubSection, SchemaPackage
from simulationworkflowschema.general import (
    SimulationWorkflow,
    SimulationWorkflowResults,
    SimulationWorkflowMethod,
)
from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure


configuration = config.get_plugin_entry_point(
    'nomad_eos_workflows.schema_packages:nomad_eos_workflows_plugin'
)

m_package = SchemaPackage()


class EOSFit(ArchiveSection):
    """
    Section containing results of an equation of state fit.
    """

    m_def = Section(validate=False)

    function_name = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the function used to perform the fitting of the volume-energy data. Value
        can be one of birch_euler, birch_lagrange, birch_murnaghan, mie_gruneisen,
        murnaghan, pack_evans_james, poirier_tarantola, tait, vinet.
        """,
    )

    fitted_energies = Quantity(
        type=np.dtype(np.float64),
        shape=['n_points'],
        unit='joule',
        description="""
        Array of the fitted energies corresponding to each volume.
        """,
    )

    bulk_modulus = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='pascal',
        description="""
        Calculated value of the bulk modulus by fitting the volume-energy data.
        """,
    )

    bulk_modulus_derivative = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Calculated value of the pressure derivative of the bulk modulus.
        """,
    )

    equilibrium_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description="""
        Calculated value of the equilibrium volume by fitting the volume-energy data.
        """,
    )

    equilibrium_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Calculated value of the equilibrium energy by fitting the volume-energy data.
        """,
    )

    rms_error = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Root-mean squared value of the error in the fitting.
        """,
    )


class EOSResults(SimulationWorkflowResults):
    volumes = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
        unit='m ** 3',
        description="""
        Array of volumes per atom for which the energies are evaluated.
        """,
    )

    energies = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
        unit='joule',
        description="""
        Array of energies corresponding to each volume.
        """,
    )

    eos_fit = SubSection(sub_section=EOSFit.m_def, repeats=True)


# ? Should this be integrated with ThermodynamicsWorkflow? It's not exactly a serialsimulation, rather a disconnected/parrallel one?
class EOSWorkflow(SimulationWorkflow, PlotSection):
    """
    A base section used to define Equation of State (EOS) workflows. These workflows are used ...
    """

    name = Quantity(
        type=str,
        default='EOS',
        description='Name of the workflow. Default set to `EOS`.',
    )

    method = SubSection(sub_section=SimulationWorkflowMethod)

    results = SubSection(sub_section=EOSResults, repeats=False)  # ? repeats?

    def extract_total_energy_differences(
        self, logger: 'BoundLogger'
    ) -> Optional[pint.Quantity]:
        """
        Extracts the total energy differences from the task outputs of the NEB workflow.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            Optional[pint.Quantity]:
        """
        # Resolve the reference of energies from the first NEB task
        ref_image_task = self.tasks[0]
        if ref_image_task.m_xpath('outputs[0].section.energy.total.value') is None:
            logger.error(
                'Could not resolve the initial value of the total energy for referencing.'
            )
            return None

        energy_reference = ref_image_task.outputs[0].section.energy.total.value.m
        energy_units = ref_image_task.outputs[0].section.energy.total.value.u

        # Append the energy differences of the images w.r.t. the reference of energies
        tot_energies = []
        for output in self.outputs:
            if output.section.energy.total.value is not None:
                tot_energies.append(
                    output.section.energy.total.value.m - energy_reference
                )
            else:
                tot_energies.append(None)  # Handle missing values safely

        # Return a pint.Quantity (list of magnitudes with associated unit)
        return tot_energies * energy_units

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        self.results = EOSResults()
        volumes = []
        energies = []
        box_deformations = []

        # if not self.method:
        #     self.method = ThermodynamicsMethod()
        #     self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        # if not self.results:
        #     self.results = ThermodynamicsResults()
        #     self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        # collect the energies, volumes, and box deformation values from each task
        for task in self.tasks:
            if not task.outputs:
                volume = None
                energy = None
                box_deformations = None
            else:
                # Extract the volume and energy values from the task outputs
                task_outputs = task.calculation[0]
                volume = task_outputs.volume or task_outputs.thermodynamics.volume or calculatie the volume from the box
                energy = task_outputs.energy.total.value
                # box_deformation = get_box_deformation() -- I guess this needs to be stripped from the file name, or is there some more robust way to do this?

        try:
            self.total_energy_differences = self.extract_total_energy_differences(
                logger=logger
            )
        except Exception:
            logger.error('Could not set NEBWorkflow.total_energy_differences.')

        # TODO - expecting one input structure...from which calc?
        # Extract system name from input structure (chemical composition of first image)
        try:
            system_name = self.tasks[0].inputs[0].section.chemical_composition_hill
        except (KeyError, IndexError, AttributeError):
            system_name = None

        # Dynamically set entry name
        archive.metadata.entry_type = 'EOS Workflow'
        if system_name is not None:
            archive.metadata.entry_name = f'{system_name} EOS Calculation'
        else:
            archive.metadata.entry_name = 'EOS Calculation'

        # ? E-V plot is already defined within the js code I believe, should this be moved to plotly?
        # Generate NEB energy plot using Plotly Express and store it in self.figures
        # try:
        #     if (
        #         self.total_energy_differences is not None
        #         and len(self.total_energy_differences) > 0
        #     ):
        #         # If energies are stored as pint.Quantity, extract magnitude and unit
        #         if hasattr(self.total_energy_differences, 'm'):
        #             magnitudes = self.total_energy_differences.m
        #             unit = str(self.total_energy_differences.u)
        #         else:
        #             magnitudes = self.total_energy_differences
        #             unit = 'eV'  # Default unit if missing

        #         # Custom unit mapping
        #         unit_mapping = {
        #             'electron_volt': 'eV',
        #             'joule': 'J',
        #             # Add more mappings as needed
        #         }

        #         # Use pint to format the unit in a pretty way
        #         ureg = pint.UnitRegistry(system='short')
        #         pretty_unit = ureg(unit).units.format_babel()
        #         pretty_unit = unit_mapping.get(
        #             pretty_unit, pretty_unit
        #         )  # Apply custom mapping

        #         logger.info(f'Formatted unit: {pretty_unit}')

        #         # Create positions as 1, 2, 3, ..., based on the number of energy entries
        #         positions = list(range(1, len(magnitudes) + 1))

        #         # Use Plotly Express to create the plot
        #         fig = px.scatter(
        #             x=positions,
        #             y=magnitudes,
        #             labels={
        #                 'x': 'Reaction Coordinates',
        #                 'y': f'Energy Difference ({pretty_unit})',
        #             },
        #         )
        #         fig.add_scatter(
        #             x=positions, y=magnitudes, mode='lines', line=dict(shape='linear')
        #         )

        #         fig.update_layout(title='NEB Energy Profile', template='plotly_white')

        #         # Convert to NOMAD-compatible PlotlyFigure
        #         self.figures.append(
        #             PlotlyFigure(label='NEB Workflow', figure=fig.to_plotly_json())
        #         )
        # except Exception as e:
        #     logger.error(f'Error while generating NEB energy plot: {e}')


m_package.__init_metainfo__()
