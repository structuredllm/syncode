from ast import Tuple
import time
import torch
import syncode.common as common
from syncode.grammar_mask.logits_processor import SyncodeLogitsProcessor
from transformers import LogitsProcessorList, StoppingCriteriaList, StoppingCriteria, PreTrainedModel
from syncode.parsers.grammars import Grammar
from syncode.utils.generation import filter_code, fix_indents
from typing import Callable, Iterable, Union
from transformers.generation.utils import GenerationMode
from transformers.generation.configuration_utils import GenerationConfig


class KeywordsStoppingCriteria(StoppingCriteria):
    '''
    Assume batch_size = 1

    We can use this class to check if the stop word is present in the completion. This is more expensive since we need to decode the completion to check if the stop word is present.
    '''
    def __init__(self, tokenizer, stop_words = []):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.stop_words_ids = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        partial_output = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        for stop_word in self.stop_words:
            if partial_output.endswith(stop_word):
                return True
        return False    


class HuggingFaceModel:
    def __init__(
            self, 
            model: Callable, 
            grammar: Grammar,
            tokenizer=None, 
            prompt_template: str = '', 
            best_of: int = 1, 
            before_prediction_hook=lambda: None, 
            device='cuda', 
            grammar_decoder=None, 
            mode: str ='original',
            opp: bool = True,
            **kwargs) -> None:
        super().__init__()

        self.prompt_template = prompt_template
        self.model: PreTrainedModel = model
        self.tokenizer = tokenizer
        self.device = self.model.device
        self.best_of = best_of
        self._before_prediction_hook = before_prediction_hook
        self.logits_processor = grammar_decoder
        self.grammar_processor: Iterable = LogitsProcessorList([self.logits_processor]) if self.logits_processor is not None else None

        self.mode = mode
        self.grammar = grammar
        self.gen_args = kwargs
        self.opp = opp

    def get_grammar_decoder(self):
        if self.grammar_processor is not None and len(self.grammar_processor) > 0:
            return self.grammar_processor[0]
        return None

    @torch.inference_mode()
    def generate_grammar_constrained_completion(
        self, 
        prompt: Union[str, list], 
        batch_size, 
        stop_words=None, 
        return_token_ids=False,
        debug=False
        ) -> Iterable[str]:
        '''
        Generates batch_size completions for the given prompt. 

        Args:
            prompt (str): The prompt for which completions are generated.
            batch_size (int): The number of completions to generate.
            stop_words (list): A list of stop words. If the completion ends with any of the stop words, the completion is returned.
            return_token_ids (bool): If True, returns the token ids of the completions.
            debug (bool): If True, prints debug information.
        '''        
        inputs = self.get_tokenized_input(prompt, batch_size)

        # Reset the grammar decoder
        if self.logits_processor is not None:
            self.logits_processor.reset()

        input_ids_cutoff = inputs.input_ids.size(dim=1)
        
        # Get the generation config
        gen_config = GenerationConfig.from_model_config(self.model.config)
        gen_config.update(**self.gen_args)

        # Get the generation mode
        gen_mode = self._get_generation_mode(gen_config)

        # Create stopping criteria
        if stop_words is not None:
            stop_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(self.tokenizer, stop_words=stop_words)])
        else:
            stop_criteria = []

        # Generate completions
        if self.opp and (gen_mode == GenerationMode.SAMPLE or gen_mode == GenerationMode.GREEDY_SEARCH) and batch_size == 1: # Use our own implementation for greedy search and sampling
            generated_ids = self._generate(
                inputs, 
                gen_config, 
                gen_mode, 
                grammar_decoder=self.logits_processor,
                stop_criteria=stop_criteria,
                debug=debug
                )
        else:
            if self.opp:
                if not (gen_mode == GenerationMode.SAMPLE or gen_mode == GenerationMode.GREEDY_SEARCH):
                    print("WARNING: Opportunistic mode requires SAMPLE or GREEDY_SEARCH generation mode.")
                if not batch_size == 1:
                    print("WARNING: Opportunistic mode requires batch_size of 1.")
            
            # Ensure pad_token_id is set
            if 'pad_token_id' not in dir(self.tokenizer):
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Use generate from transformers library for other modes
            generated_ids = self.model.generate(
                **inputs, 
                logits_processor=self.grammar_processor, 
                stop_strings=stop_words,
                tokenizer=self.tokenizer,
                **self.gen_args
                )
        batch_completions = []

        # Total tokens generated
        self.total_tokens = 0

        # Decode the completions
        for i in range(batch_size):
            self.total_tokens += len(generated_ids[i])-input_ids_cutoff+1
            completion = self.tokenizer.decode(generated_ids[i][input_ids_cutoff:len(generated_ids[i])], skip_special_tokens=True)

            if return_token_ids:
                batch_completions.append((completion, generated_ids[i], inputs.input_ids))
            else:                                       
                batch_completions.append(completion)
        
        return batch_completions


    def get_tokenized_input(self, prompt: Union[str, list], batch_size: int):
        """
        Tokenizes the input prompt and returns the input dictionary for the model.
        """            
        if isinstance(prompt, list):
            assert 'apply_chat_template' in dir(self.tokenizer), "Tokenizer does not support chat completion"
            prompt_str = self.tokenizer.apply_chat_template(
                prompt, 
                add_generation_prompt=True, 
                return_tensors="pt",
                tokenize=False
                )
        elif isinstance(prompt, str):
            prompt_str = prompt
        else:
            raise ValueError("Prompt should be either a string or a list! It is currently of type: "+str(type(prompt)))

        input_batch = [prompt_str for _ in range(batch_size)]
        inputs = self.tokenizer(input_batch, return_tensors="pt").to(self.model.device)

        return inputs

    @torch.inference_mode()
    def _generate(
        self, 
        inputs:dict, 
        gen_config:GenerationConfig, 
        gen_mode:GenerationMode, 
        grammar_decoder:SyncodeLogitsProcessor=None,
        stop_criteria:StoppingCriteria=[],
        debug=False
        ):
        """
        We support greedy search and sampling for batch size 1 otherwise we use the generate function from transformers library.
        """
        token_ids, attention_mask, past_key_values = inputs['input_ids'], inputs['attention_mask'], None
        
        # This does not include grammar decoder
        self.model._prepare_special_tokens(gen_config, False, device=self.device)

        # Add logits processor for generation parameters such as top_k, top_p, temperature, etc.
        logits_processor = self.model._get_logits_warper(gen_config, self.device)

        max_tokens = self.gen_args['max_new_tokens']+token_ids.size(1)
        self.model.config.pad_token_id = pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        while True:
            try:
                if past_key_values: # Get the last token if kv is cached for all previous tokens
                    input_ids = token_ids[..., -1].unsqueeze(-1) 
                else:
                    input_ids = token_ids

                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    past_key_values=past_key_values
                    )                
            except IndexError as e:  
                raise ValueError(f"The input length exceeds the context length of the model. {e}")

            next_token_scores, past_key_values = outputs.logits[:, -1, :], outputs.past_key_values
            
            if grammar_decoder is not None:
                next_token = self._get_next_token(gen_mode, token_ids, logits_processor, next_token_scores)
                is_valid = grammar_decoder.is_valid(token_ids, next_token)

                if not is_valid:
                    # calling grammar decoder is expensive. Hence, in the opportunist mode, we call it only when the standard generation is syntactically incorrect
                    next_token_scores = grammar_decoder(token_ids, next_token_scores)
                    next_token = self._get_next_token(gen_mode, token_ids, logits_processor, next_token_scores)
            else:
                next_token = self._get_next_token(gen_mode, token_ids, logits_processor, next_token_scores)
            
            token_ids = torch.cat([token_ids, next_token[:, None]], dim=-1)

            # Check stopping criteria
            finish_generation = False
            for stop_criterion in stop_criteria:
                if stop_criterion(token_ids, next_token_scores):
                    finish_generation = True
                    
             # Check if the next token is the end of the sequence or the max tokens is reached
            if finish_generation or next_token == self.tokenizer.eos_token_id or token_ids.size(1) >= max_tokens:
                break

            # Update attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype).to(self.device)], dim=-1)
        
        if debug:
            grammar_decoder.print_debug()

        return token_ids

    def _get_next_token(self, gen_mode, token_ids, logits_processor, next_token_scores):
        if gen_mode == GenerationMode.GREEDY_SEARCH:
            next_token = torch.argmax(next_token_scores, dim=-1)
        elif gen_mode == GenerationMode.SAMPLE:
            next_token_scores = logits_processor(token_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_token

    def _get_generation_mode(
        self, generation_config: GenerationConfig
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by a [`GenerationConfig`] instance.
        """
        if generation_config.constraints is not None or generation_config.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif generation_config.num_beams == 1:
            if generation_config.do_sample is False:
                if (
                    generation_config.top_k is not None
                    and generation_config.top_k > 1
                    and generation_config.penalty_alpha is not None
                    and generation_config.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if generation_config.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif generation_config.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH
        return generation_mode
    
    def tokenize(self, s: str) -> 'Iterable[int]':
        return self.tokenizer.encode(s, add_special_tokens=False)
