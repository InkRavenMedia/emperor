"""Lossless byte generation with learned boundaries and explicit truncation."""

import codecs

import torch

from emperor.embedding.hierarchical import (
    BYTE_VALUES,
    END_OF_DOCUMENT,
    END_OF_TOKEN,
    OUTPUT_SYMBOLS,
)
from emperor.nn import freeze_forward_schedules

from ._records import (
    ByteTokenGenerationOutput,
)
from ._validation import HierarchicalByteGenerationValidator


class HierarchicalByteGenerator:
    VALIDATOR = HierarchicalByteGenerationValidator

    def __init__(self, model):
        self.model = model

    def generate(self, conditioning, prefix, max_new_bytes, options, generator):
        self.VALIDATOR.validate_options(options)
        prefix_bytes = self.VALIDATOR.validate_and_encode_prefix(
            prefix, self.model.cfg.max_token_bytes
        )
        if max_new_bytes is None:
            max_new_bytes = self.model.cfg.max_token_bytes
        self.VALIDATOR.validate_byte_budget(max_new_bytes)
        self.VALIDATOR.validate_conditioning(
            conditioning, self.model.cfg.conditioning_dim
        )
        if generator is None and options.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(options.seed)
        training_states = [(child, child.training) for child in self.model.modules()]
        try:
            self.model.eval()
            with torch.no_grad(), freeze_forward_schedules():
                return self.__generate_bytes(
                    conditioning, prefix_bytes, max_new_bytes, options, generator
                )
        finally:
            for child, was_training in training_states:
                child.training = was_training

    def __generate_bytes(
        self, conditioning, token_bytes, max_new_bytes, options, generator
    ):
        initial_byte_count = len(token_bytes)
        requested_byte_limit = initial_byte_count + max_new_bytes
        byte_limit = min(self.model.cfg.max_token_bytes, requested_byte_limit)
        stop_reason = (
            "byte_limit"
            if byte_limit == self.model.cfg.max_token_bytes
            else "byte_budget"
        )
        if max_new_bytes == 0:
            return ByteTokenGenerationOutput(bytes(token_bytes), "byte_budget", 0)
        while True:
            next_symbol_logits = self.__next_symbol_logits(conditioning, token_bytes)
            current_byte_count = len(token_bytes)
            remaining_bytes = byte_limit - current_byte_count
            # At the cap, permit every syntactically valid next byte to compete
            # with controls. UTF-8 needs at most four bytes; a cap must not force EOW.
            utf8_budget = remaining_bytes or 4
            allowed_symbols = (
                self.__allowed_symbols(token_bytes, utf8_budget)
                if options.constrain_utf8
                else torch.ones(OUTPUT_SYMBOLS, dtype=torch.bool)
            )
            if not bool(allowed_symbols.any()):
                break
            sampling_generator = (
                generator if generator is not None else torch.default_generator
            )
            generator_state = (
                sampling_generator.get_state() if options.do_sample else None
            )
            selected_symbol = self.__select_symbol(
                next_symbol_logits, allowed_symbols, options, generator
            )
            if selected_symbol in (END_OF_TOKEN, END_OF_DOCUMENT):
                stop_reason = (
                    "end_of_token"
                    if selected_symbol == END_OF_TOKEN
                    else "end_of_document"
                )
                break
            if current_byte_count == byte_limit:
                # A non-control prediction cannot turn this partial word into a
                # completed word. Preserve its draw for a resumed byte budget.
                if generator_state is not None:
                    sampling_generator.set_state(generator_state)
                break
            token_bytes.append(selected_symbol)
        final_byte_count = len(token_bytes)
        new_byte_count = final_byte_count - initial_byte_count
        return ByteTokenGenerationOutput(
            bytes(token_bytes), stop_reason, new_byte_count
        )

    def __next_symbol_logits(self, conditioning, token_bytes):
        prefix_values = list(token_bytes)
        prefix_ids = torch.tensor(
            prefix_values, dtype=torch.long, device=conditioning.device
        )
        batched_prefix_ids = prefix_ids.reshape(1, 1, -1)
        prefix_byte_count = len(token_bytes)
        byte_lengths = batched_prefix_ids.new_tensor([[prefix_byte_count]])
        batched_conditioning = conditioning.reshape(1, 1, -1)
        predictions = self.model(batched_conditioning, batched_prefix_ids, byte_lengths)
        return predictions.logits[-1]

    def __allowed_symbols(self, token_bytes, remaining_bytes):
        prefix_decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            prefix_decoder.decode(token_bytes, final=False)
        except UnicodeDecodeError as error:
            raise ValueError(
                "constrain_utf8 requires a valid or incomplete UTF-8 prefix"
            ) from error
        pending_bytes = prefix_decoder.getstate()[0]
        allowed_symbols = torch.zeros(OUTPUT_SYMBOLS, dtype=torch.bool)
        for byte_value in range(BYTE_VALUES):
            candidate_decoder = codecs.getincrementaldecoder("utf-8")()
            candidate_byte = bytes([byte_value])
            candidate_bytes = pending_bytes + candidate_byte
            try:
                candidate_decoder.decode(candidate_bytes, final=False)
            except UnicodeDecodeError:
                continue
            incomplete_character = candidate_decoder.getstate()[0]
            required_bytes = self.__required_continuation_bytes(incomplete_character)
            allowed_symbols[byte_value] = required_bytes < remaining_bytes
        allowed_symbols[END_OF_TOKEN] = bool(token_bytes) and not pending_bytes
        allowed_symbols[END_OF_DOCUMENT] = not token_bytes
        return allowed_symbols

    def __required_continuation_bytes(self, incomplete_character):
        if not incomplete_character:
            return 0
        leading_byte = incomplete_character[0]
        character_width = 2 if leading_byte < 224 else 3 if leading_byte < 240 else 4
        pending_byte_count = len(incomplete_character)
        return character_width - pending_byte_count

    def __select_symbol(self, logits, allowed_symbols, options, generator):
        sampling_logits = logits.detach().float().cpu()
        self.VALIDATOR.validate_logits(sampling_logits)
        scaled_logits = sampling_logits / options.temperature
        filtered_logits = scaled_logits.masked_fill(~allowed_symbols, -torch.inf)
        if not options.do_sample:
            selected_index = filtered_logits.argmax()
            return int(selected_index)
        if options.top_k is not None:
            top_scores = filtered_logits.topk(options.top_k).values
            minimum_top_score = top_scores[-1]
            below_top_k = filtered_logits < minimum_top_score
            filtered_logits = filtered_logits.masked_fill(below_top_k, -torch.inf)
        if options.top_p < 1:
            filtered_logits = self.__filter_nucleus_logits(
                filtered_logits, options.top_p
            )
        symbol_probabilities = filtered_logits.softmax(-1)
        sampled_index = torch.multinomial(symbol_probabilities, 1, generator=generator)
        return int(sampled_index)

    def __filter_nucleus_logits(self, logits, top_p):
        ordered_logits, symbol_order = logits.sort(descending=True)
        ordered_probabilities = ordered_logits.softmax(-1)
        cumulative_probabilities = ordered_probabilities.cumsum(-1)
        preceding_probabilities = cumulative_probabilities - ordered_probabilities
        excluded_symbols = preceding_probabilities >= top_p
        excluded_indices = symbol_order[excluded_symbols]
        logits[excluded_indices] = -torch.inf
        return logits
