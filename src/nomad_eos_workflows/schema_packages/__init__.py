from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class NOMADEOSWorkflowsEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_eos_workflows.schema_packages.eos import m_package

        return m_package


nomad_eos_workflows_plugin = NOMADEOSWorkflowsEntryPoint(
    name='NOMADEOSWorkflows',
    description='Schema package plugin for the NOMAD EOS workflows definitions.',
)
