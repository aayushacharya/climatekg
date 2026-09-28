import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import ontology_conformance_agent as agent


ONTOLOGY = {
    ("Source", "SourceComponent", "HAS_SOURCE_COMPONENT"),
    ("Source", "PhysicalScheme", "USES_PHYSICAL_SCHEME"),
    ("Source", "PhysicalFeature", "HAS_PHYSICAL_FEATURE"),
}


class OntologyConformanceBoundaryTest(unittest.TestCase):
    def assert_conforms(self, triplet, expected):
        self.assertIs(agent.triplet_conforms(triplet, ONTOLOGY), expected)

    def test_rejects_realm_as_source_component(self):
        self.assert_conforms(
            {
                "source_entity_type": "Source",
                "target_entity_type": "SourceComponent",
                "relation": "HAS_SOURCE_COMPONENT",
                "target_entity_instance": "land",
            },
            False,
        )

    def test_accepts_named_source_component(self):
        self.assert_conforms(
            {
                "source_entity_type": "Source",
                "target_entity_type": "SourceComponent",
                "relation": "HAS_SOURCE_COMPONENT",
                "target_entity_instance": "JSBACH",
            },
            True,
        )

    def test_rejects_process_name_without_physical_scheme_cue(self):
        self.assert_conforms(
            {
                "source_entity_type": "Source",
                "target_entity_type": "PhysicalScheme",
                "relation": "USES_PHYSICAL_SCHEME",
                "target_entity_instance": "convection",
                "paragraph": "The model simulates convection.",
            },
            False,
        )

    def test_accepts_named_physical_scheme_with_cue(self):
        self.assert_conforms(
            {
                "source_entity_type": "Source",
                "target_entity_type": "PhysicalScheme",
                "relation": "USES_PHYSICAL_SCHEME",
                "target_entity_instance": "Tiedtke convection parameterization",
                "paragraph": "The model uses the Tiedtke convection parameterization.",
            },
            True,
        )

    def test_rejects_variable_as_physical_feature(self):
        self.assert_conforms(
            {
                "source_entity_type": "Source",
                "target_entity_type": "PhysicalFeature",
                "relation": "HAS_PHYSICAL_FEATURE",
                "target_entity_instance": "tas",
            },
            False,
        )

    def test_rejects_nominal_resolution_as_physical_feature(self):
        self.assert_conforms(
            {
                "source_entity_type": "Source",
                "target_entity_type": "PhysicalFeature",
                "relation": "HAS_PHYSICAL_FEATURE",
                "target_entity_instance": "nominal resolution: 1 deg",
            },
            False,
        )

    def test_accepts_grid_architecture_as_physical_feature(self):
        self.assert_conforms(
            {
                "source_entity_type": "Source",
                "target_entity_type": "PhysicalFeature",
                "relation": "HAS_PHYSICAL_FEATURE",
                "target_entity_instance": "tripolar grid",
            },
            True,
        )


if __name__ == "__main__":
    unittest.main()
