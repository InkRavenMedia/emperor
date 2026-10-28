from emperor.augmentations.adaptive_parameters._decay.base import DecayAbstract


class DecayValidator:
    @staticmethod
    def validate_decay_variant(
        decay_variant: DecayAbstract | None,
        decay_schedule_option: object,
    ) -> DecayAbstract:
        if decay_variant is None:
            raise ValueError(
                f"Unsupported decay_schedule value: {decay_schedule_option!r}."
            )
        return decay_variant
