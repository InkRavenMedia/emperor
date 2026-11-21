import unittest

from emperor.sampler._selection.base import SamplerBase
from emperor.sampler._selection.validation import SamplerBaseValidator
from support.sampler import sampler_config as make_config


class TestSamplerBaseValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(SamplerBase.VALIDATOR, SamplerBaseValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(SamplerBaseValidator):
            @staticmethod
            def _validate_num_topk_samples(num_topk_samples, top_k):
                raise RuntimeError("substituted construction validator was called")

        class TrackingSampler(SamplerBase):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingSampler(make_config())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(SamplerBaseValidator):
            @staticmethod
            def _validate_router_logit_scores(model, router_logit_scores):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingSampler(SamplerBase):
            VALIDATOR = RejectingValidator

        model = RejectingSampler(make_config())

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model.get_probabilities_and_indices([])

    def test_positive_integer_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "top_k must be a positive integer, received 0",
        ):
            SamplerBase(make_config(top_k=0))


if __name__ == "__main__":
    unittest.main()
