import unittest

from emperor.sampler._selection.top_k import SamplerTopk
from emperor.sampler._selection.validation import SamplerTopkValidator
from support.sampler import sampler_config as make_config


class TestSamplerTopkValidatorAdapter(unittest.TestCase):
    def test_module_exposes_specialized_validator_adapter(self):
        self.assertIs(SamplerTopk.VALIDATOR, SamplerTopkValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(SamplerTopkValidator):
            @staticmethod
            def _validate_topk_configuration(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingSamplerTopk(SamplerTopk):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingSamplerTopk(make_config())

    def test_topk_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "top_k must be greater than 0 and less than num_experts when using "
            "SamplerTopk, received top_k=4, num_experts=4",
        ):
            SamplerTopk(make_config(top_k=4))


if __name__ == "__main__":
    unittest.main()
