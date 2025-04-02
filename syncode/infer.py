import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import fire
import torch
from typing import Optional, Literal, Union

import syncode.common as common
from syncode.language_model import HuggingFaceModel
from syncode.grammar_mask.logits_processor import SyncodeLogitsProcessor
from syncode.parsers.grammars import Grammar
from syncode.dataset import Dataset
from syncode.evaluation.code_eval import CodeEval
from syncode.evaluation.math_eval import MathEval
from syncode.evaluation.sql_eval import SQLEval
from syncode.evaluation.json_eval import JSONEval
from syncode.evaluation.fol_eval import FOLEval


class Syncode:
    """Syncode class for running inference on a model with optional grammar-based constraints.
    
    Args:
        model (str): Model id for huggingface model hub or model name if stored locally.
        mode (str, optional): Mode for inference. Defaults to "grammar_strict". 
            "original" for original model, "grammar_mask" for grammar mask, "grammar_strict" for strict grammar mask.
        quantize (bool, optional): Quantize model. Defaults to True.
        device (str, optional): Device to use. Defaults to "cuda".
        grammar (str, optional): Language to use for grammar constraints. Defaults to None.
            Options include "python", "go", "calc", "sql", "json", "fol".
        parse_output_only (bool, optional): Parse only the output. Defaults to True.
        dev_mode (bool, optional): Development mode. Defaults to False.
        new_mask_store (bool, optional): Use new DFA mask store. Defaults to False.
        parser (str, optional): Parser to use. Defaults to "lalr". Options are "lr" and "lalr".
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        opp (bool, optional): Whether to use opportunistic generation. Defaults to True.
        **kwargs: Additional arguments passed to the model for generation.
    """
    def __init__(
        self, 
        model: str,
        mode: Literal["original", "grammar_mask", "grammar_strict"] = "grammar_strict",
        quantize: bool = True,
        device: str = "cuda",
        grammar: Optional[str] = None,
        parse_output_only: bool = True,
        dev_mode: bool = False,
        new_mask_store: bool = False,
        parser: Literal["lr", "lalr"] = "lalr",
        seed: Optional[int] = None,
        opp: bool = True,
        device_map: Optional[str] = None,
        **kwargs
    ):  
        # Check inputs
        valid_modes = ["original", "grammar_mask", "grammar_strict"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")
            
        gen_kwargs = {'max_length', 'max_new_tokens', 'min_length', 'min_new_tokens', 
                     'early_stopping', 'do_sample', 'num_beams', 'use_cache', 
                     'temperature', 'top_k', 'top_p', 'num_return_sequences', 
                     'pad_token_id', 'eos_token_id'}
        
        invalid_kwargs = set(kwargs.keys()) - gen_kwargs
        if invalid_kwargs:
            raise ValueError(f"Invalid arguments: {invalid_kwargs}")

        # Set attributes
        self.mode = mode
        self.model_name = model
        self.quantize = quantize
        self.device = device
        self.num_samples = kwargs.get('num_return_sequences', 1)
        self.new_mask_store = new_mask_store
        self.parser = parser
        self.parse_output_only = parse_output_only

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Set the grammar
        self.language = grammar
        self.grammar = Grammar(grammar) if self._is_grammar_mode() else None

        # Load model and tokenizer
        model = common.load_model(self.model_name, device, quantize, device_map)
        tokenizer = common.load_tokenizer(self.model_name)
        
        # Initialize grammar decoder if needed
        self.grammar_decoder = None
        
        if self._is_grammar_mode():
            self.grammar_decoder = SyncodeLogitsProcessor(
                self.grammar, 
                tokenizer=tokenizer, 
                use_cache=(not self.new_mask_store), 
                parse_output_only=self.parse_output_only,
                num_samples=self.num_samples, 
                dev_mode=dev_mode,
                parser=parser,
                mode=mode,
            )

        # Set default max new tokens if not provided
        kwargs['max_new_tokens'] = kwargs.get('max_new_tokens', 200)

        # Initialize the model
        self.model = HuggingFaceModel(
            model, 
            grammar=self.grammar,
            tokenizer=tokenizer, 
            device=device, 
            grammar_decoder=self.grammar_decoder, 
            mode=self.mode,
            opp=opp,
            **kwargs
        )

    def _is_grammar_mode(self):
        """Check if using a grammar-based mode."""
        return self.mode in ['grammar_mask', 'grammar_strict']

    def infer(self, prompt=None, stop_words=None, debug=False):
        """Run inference with an optional prompt.
        
        Args:
            prompt (str or list, optional): The input prompt. If None, will enter interactive mode.
            stop_words (list, optional): Words that signal the end of generation.
            debug (bool, optional): Enable debug output. Defaults to False.
            
        Returns:
            str: The generated output for the provided prompt.
        """
        return self.user_input(prompt, stop_words=stop_words, debug=debug)

    def user_input(self, prompt: Union[str, list, None], stop_words=None, debug=False):
        """Process user input for inference.
        
        Args:
            prompt (str, list, or None): The input prompt. If None, enters interactive mode.
            stop_words (list, optional): Words that signal the end of generation.
            debug (bool, optional): Enable debug output. Defaults to False.
            
        Returns:
            str: The generated output.
        """
        if prompt:
            if isinstance(prompt, list) and not self.parse_output_only:
                raise ValueError("Prompt must be a string for input+output parsing")

            return self.model.generate_grammar_constrained_completion(
                prompt, self.num_samples, stop_words=stop_words, debug=debug
            )
        else:
            # Interactive mode
            while True:
                prompt = input('Enter prompt: ')
                
                # Handle escape sequences for Python grammar
                if self.grammar and self.grammar.name == 'python':
                    prompt = prompt.replace('\\n', '\n').replace('\\"', '\"')\
                                  .replace('\\t', '\t').replace("\\'", "\'") \
                                  .replace('\\b', '\b').replace('\\r', '\r')
                                  
                if prompt == "exit":
                    break

                batch_completions = self.model.generate_grammar_constrained_completion(
                    prompt, self.num_samples
                )
                for completion in batch_completions:
                    print(prompt + completion)

    def evaluate(
            self, 
            dataset: Literal["mbxp", "humaneval", "mathqa-x", "gsm8k", "spider", "json_eval"],
            out_path: Optional[str] = None,
            num_tasks: Optional[int] = None,
            num_few_shot: int = 0,
            logger = None,
            task_id: Optional[int] = None,
            prompt_type: str = 'original',  # For JSONEval: "original" or "explicit"
            format_tabs: bool = False  # For CodeEval: Format tabs in prompt
        ) -> dict:
        """Run evaluation on the model with the specified dataset.

        Args:
            dataset (str): Dataset to evaluate on (e.g., "mbxp", "humaneval", "mathqa-x").
            out_path (str, optional): Output path for evaluation results.
            num_tasks (int, optional): Number of tasks to evaluate.
            num_few_shot (int, optional): Number of examples for few-shot prompting.
            logger (Logger, optional): Logger instance for logging evaluation progress.
            task_id (int, optional): For debugging a specific task.
            prompt_type (str, optional): Prompt type for JSONEval ("original" or "explicit").
            format_tabs (bool, optional): Format tabs in prompt for CodeEval.
            
        Returns:
            dict: Evaluation results.
        """
        # Use empty logger if none provided
        if logger is None:
            logger = common.EmptyLogger()
            
        if logger.is_closed:
            logger.open()

        # Load the dataset
        self.dataset = Dataset(dataset, language=self.language, num_few_shot=num_few_shot)

        # Run the appropriate evaluation based on dataset type
        if self.dataset.type == "code":
            output = CodeEval.run_code_eval(
                self, self.num_samples, out_path, format_tabs=format_tabs, 
                debug_task_id=task_id, logger=logger, num_tasks=num_tasks
            )
        elif self.dataset.type == "math":
            output = MathEval.run_math_eval(
                self, out_path, debug_task_id=task_id, logger=logger
            )
        elif self.dataset.type == "sql":
            output = SQLEval.run_eval(
                self, out_path, debug_task_id=task_id, num_tasks=num_tasks
            )
        elif self.dataset.type == "fol":
            output = FOLEval.run_eval(
                self, out_path, debug_task_id=task_id
            )
        elif self.dataset.type == "json":
            output = JSONEval.run_json_eval(
                self, out_path, debug_task_id=task_id, logger=logger, 
                prompt_type=prompt_type, num_tasks=num_tasks
            )
        else:
            raise ValueError(f"Dataset type {self.dataset.type} not supported")
            
        logger.close()
        return output


def main(
    model: str,
    mode: str = "grammar_mask",
    dataset: str = "input",
    grammar: Optional[str] = None,
    quantize: bool = True,
    device: str = "cuda",
    num_few_shot: int = 0,
    dev_mode: bool = False,
    log_level: int = 2, # Default to info level
    new_mask_store: bool = False,
    parser: str = "lalr",
    num_tasks: Optional[int] = None,
    task_id: Optional[int] = None,
    seed: Optional[int] = None,
    opp: bool = True,
    debug: bool = False,
    parse_output_only: bool = True,
    prompt_type: str = 'original',
    format_tabs: bool = False,
    device_map: Optional[str] = None,
    **kwargs
):
    """Run Syncode with the specified configuration.
    
    Args:
        model (str): Model ID for HuggingFace model hub or path to local model.
        mode (str, optional): Inference mode. Defaults to "grammar_strict".
        dataset (str, optional): Dataset to evaluate or "input" for interactive mode.
        grammar (str, optional): Grammar to use. Defaults to None.
        quantize (bool, optional): Whether to quantize the model. Defaults to True.
        device (str, optional): Device to run on. Defaults to "cuda".
        num_few_shot (int, optional): Number of few-shot examples. Defaults to 0.
        dev_mode (bool, optional): Enable development mode. Defaults to False.
        log_level (int, optional): Logging level. Defaults to 1.
        new_mask_store (bool, optional): Use new DFA mask store. Defaults to False.
        parser (str, optional): Parser to use. Defaults to "lalr".
        num_tasks (int, optional): Number of tasks to evaluate. Defaults to None.
        task_id (int, optional): Specific task ID to evaluate. Defaults to None.
        seed (int, optional): Random seed. Defaults to None.
        opp (bool, optional): Use opportunistic generation. Defaults to True.
        debug (bool, optional): Enable debug mode. Defaults to False.
        parse_output_only (bool, optional): Parse only the output. Defaults to True.
        prompt_type (str, optional): Prompt type for JSONEval. Defaults to 'original'.
        format_tabs (bool, optional): Format tabs for CodeEval. Defaults to False.
        **kwargs: Additional arguments for model generation.
    """
    # Map log_level parameter to standard logging levels
    logging_levels = {
        0: logging.ERROR,     # Only errors
        1: logging.WARNING,   # Warnings and errors (default)
        2: logging.INFO,      # Info, warnings, and errors
        3: logging.DEBUG      # All debug information
    }
    
    # Set the log level for the root logger
    logging.getLogger().setLevel(logging_levels.get(log_level, logging.WARNING))
    
    # Initialize Syncode
    syncode = Syncode(
        model=model, 
        mode=mode, 
        quantize=quantize, 
        device=device, 
        grammar=grammar, 
        dev_mode=dev_mode, 
        new_mask_store=new_mask_store, 
        parser=parser, 
        seed=seed, 
        opp=opp,
        parse_output_only=parse_output_only,
        device_map=device_map,
        **kwargs
    )
    
    if dataset == "input":
        # Interactive mode
        syncode.infer(debug=debug)
    else:
        # Evaluation mode
        # Setup output directory and logger
        num_samples = kwargs.get('num_return_sequences', 1)
        out_dir, out_path = common.get_output_path(model, grammar, dataset, num_samples, mode)
        logger = common.Logger(num_samples, mode, parser, out_dir, log_level=log_level, task_id=task_id)
        
        if syncode.grammar_decoder is not None:
            syncode.grammar_decoder.logger = logger

        # Run evaluation
        syncode.evaluate(
            dataset=dataset, 
            num_tasks=num_tasks, 
            task_id=task_id, 
            out_path=out_path, 
            logger=logger, 
            num_few_shot=num_few_shot,
            prompt_type=prompt_type,
            format_tabs=format_tabs
        )


if __name__ == "__main__":
    fire.Fire(main)
