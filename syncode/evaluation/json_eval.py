from tqdm import tqdm
from typing import Optional
from syncode.evaluation.mxeval.data import write_jsonl
import ast
import json
from jsonschema import validate, ValidationError
from collections import defaultdict
from syncode import common
from syncode.evaluation.mxeval_evaluation import compute_pass_at_k

class JSONEval:
    """
    Run evaluation for the code generation model
    """
    @staticmethod
    def run_json_eval(
        syncode,
        out_path: Optional[str], 
        debug_task_id: Optional[int] = None,
        logger=common.EmptyLogger(),
        prompt_type='original',
        num_tasks=None
        ):
        problems = syncode.dataset.problems
        if syncode.grammar_decoder is not None:
            # For this evaluation, we only parse the output (and not the input+output)
            syncode.grammar_decoder.parse_output_only = True

        if debug_task_id is not None:
            problems = [problems[debug_task_id]]
        
        if num_tasks is not None:
            problems = problems[:num_tasks]

        samples = []
        outputs = []
        pbar = tqdm(total=len(problems) * syncode.num_samples)
        results = defaultdict(list)
        
        for task_id, problem in enumerate(problems):
            output = JSONEval.run_eval_for_task(syncode, syncode.num_samples, problem, samples, pbar, task_id, prompt_type=prompt_type)
            
            if debug_task_id is not None:
                # This is for debugging purposes
                logger.log_code(msg="Problem", code=output[0])
                return output
            
            outputs.append(outputs) 

        schema_result = validate_json_data(syncode, samples, results, logger=logger)

        # exact match evaluation doesn't make sense
        # exact_match_result = validate_json_completion(samples, results, logger=logger)
        if out_path is not None:
            write_jsonl(out_path, schema_result)
        
        pass_at_k = compute_pass_at_k(results)

        # Log result
        logger.log(f"Result: {pass_at_k}")
        print(f"Result: {pass_at_k}")
        logger.close()
        return outputs
    
    def run_eval_for_task(syncode, num_samples_per_task, problem, samples, pbar, task_id, prompt_type='original'):
        """
        run evaluation for a specific task
        """
        
        if prompt_type == 'original':
            problem["prompt"][0]['content'] = f"{problem['prompt'][0]['content']}\nJSON:\n"
        else:
            problem["prompt"][0]['content'] = f"{problem['prompt'][0]['content']}\nOnly output JSON.\nJSON:\n"

        if syncode.model.tokenizer.chat_template is not None:
            prompt = syncode.model.tokenizer.apply_chat_template(problem["prompt"], tokenize = False)
        else:
            prompt = problem["prompt"][0]['content']

        batch_completions = syncode.model.generate_grammar_constrained_completion(prompt, num_samples_per_task)
        for completion_id, completion in enumerate(batch_completions):
            result = dict(
                    task_id = task_id, 
                    completion_id = completion_id,
                    completion=completion, 
                    ground_truth = problem["ground_truth"], 
                    schema = problem["schema"]
                )
            samples += [result]
        pbar.update(num_samples_per_task)
        return batch_completions
    
def validate_json_data(syncode, samples, results, logger=common.EmptyLogger()):
    validation_results = []
    for sample in samples:
        valid = False
        error_message = None
        generated_json = None
        error_type = "Schema Validation Error"
        
        json_object = sample["completion"]
        json_schema = json.loads(sample["schema"])
        
        try:
            try:
                generated_json = json.loads(json_object)
            except json.decoder.JSONDecodeError:
                try:
                    generated_json = ast.literal_eval(json_object)
                except (SyntaxError, ValueError) as e:
                    generated_json = json_object
                    error_message = f"JSON decoding error: {e}"
                    logger.log(f"Validation failed for JSON data: {error_message}")
                    validation_results.append(dict(task_id = sample["task_id"], completion = generated_json, schema = json_schema, ground_truth = None, result = error_message, passed = valid, error_type = "Decoding Error"))
                    results[sample["task_id"]].append((sample["completion_id"], dict(task_id= sample["task_id"], completion= sample["completion"], passed= valid)))
                    continue

            if isinstance(generated_json, list):
                for index, item in enumerate(generated_json):
                    try:
                        validate(instance=item, schema=json_schema)
                        logger.log(f"Item {index+1} is valid against the schema.")
                    except ValidationError as e:
                        error_message = f"Validation failed for item {index+1}: {e}"
                        break
            else: 
                try:
                    validate(instance=generated_json, schema=json_schema)
                except ValidationError as e:
                    error_message = f"Validation failed: {e}"
        except Exception as e:
            error_message = f"Error occurred: {e}"
            error_type = "Other Error"

        if error_message is None:
            valid = True
            error_message = "JSON data is valid against the schema."
            error_type = "No Error"
    
        validation_results.append(dict(task_id = sample["task_id"], completion = generated_json, schema = json_schema, ground_truth = None, result = error_message, passed = valid, error_type = error_type))
        results[sample["task_id"]].append((sample["completion_id"], dict(task_id= sample["task_id"], completion= sample["completion"], passed= valid)))

    return validation_results

def validate_json_completion(samples, results, logger=common.EmptyLogger()):
    validation_results = []
    for sample in samples:
        json_object = sample["completion"]
        ground_truth_json = json.loads(sample["ground_truth"])
        valid = False
        value_flag = False
        try:
            try:
                generated_json = json.loads(json_object)
            except json.decoder.JSONDecodeError:
                try:
                    generated_json = ast.literal_eval(json_object)
                except (SyntaxError, ValueError) as e:
                    generated_json = json_object
                    error_message = f"JSON decoding error: {e}"
                    logger.log(f"Validation failed for JSON data: {error_message}")
                    validation_results.append(dict(task_id = sample["task_id"], completion = generated_json, schema = None, ground_truth = ground_truth_json, result = error_message, passed = valid, error_type = "Decoding Error"))
                    results[sample["task_id"]].append((sample["completion_id"], dict(task_id= sample["task_id"], completion= sample["completion"], passed= valid)))
                    continue
            
            
            if set(generated_json.keys()) != set(ground_truth_json.keys()):
                logger.log("Keys don't match:")
                logger.log(f"Expected: {set(generated_json.keys())}")
                logger.log(f"Got: {set(ground_truth_json.keys())}")
                error_message = f"Keys don't match: Expected: {set(generated_json.keys())} Got: {set(ground_truth_json.keys())}"
                validation_results.append(dict(task_id = sample["task_id"], completion = generated_json, schema = None, ground_truth = ground_truth_json, result = error_message, passed = valid, error_type = "Key Match Error"))
                results[sample["task_id"]].append((sample["completion_id"], dict(task_id= sample["task_id"], completion= sample["completion"], passed= valid)))
                continue

            for key in generated_json.keys():
                if generated_json[key] != ground_truth_json[key]:
                    logger.log(f"Values don't match for key '{key}'")
                    logger.log(f"Expected: {generated_json[key]}")
                    logger.log(f"Got: {ground_truth_json[key]}")
                    error_message = f"Values don't match for key '{key}' Expected: {generated_json[key]} Got: {ground_truth_json[key]}"
                    validation_results.append(dict(task_id = sample["task_id"], completion = generated_json, schema = None, ground_truth = ground_truth_json, result = error_message, passed = valid, error_type = "Value Match Error"))
                    results[sample["task_id"]].append((sample["completion_id"], dict(task_id= sample["task_id"], completion= sample["completion"], passed= valid)))
                    value_flag = True
                    break
            if value_flag:
                continue
        
        except Exception as e:
            logger.log(f"Exception occured: {e}")
            error_message = f"Error occurred: {e}"
            error_type = "Other Error"
            validation_results.append(dict(task_id = sample["task_id"], completion = generated_json, schema = None, ground_truth = ground_truth_json, result = error_message, passed = valid, error_type = error_type))
            results[sample["task_id"]].append((sample["completion_id"], dict(task_id= sample["task_id"], completion= sample["completion"], passed= valid)))
            continue
        
        validation_results.append(dict(task_id = sample["task_id"], completion = generated_json, schema = None, ground_truth = ground_truth_json, result = "JSON data is an exact match", passed = True, error_type = "No Error"))
        results[sample["task_id"]].append((sample["completion_id"], dict(task_id= sample["task_id"], completion= sample["completion"], passed= True)))
    return validation_results