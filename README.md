# ToM Reasoning Evaluation for LLMs

This repository contains scripts to process and evaluate large language models (LLMs) on theory of mind (ToM) reasoning tasks. The dataset consists of stories involving agents, objects, and locations, with questions testing different aspects of ToM reasoning such as false belief and memory.

## Data Structure

The dataset is split into train, validation, and test sets. Each set consists of:
- A `.txt` file containing stories and questions
- A `.trace` file containing metadata and question types

Each story is numbered, with a question at the end. The question types in the trace files include:
- `memory`: Questions about remembering information from the story
- `reality`: Questions about the current state of the world
- `first_order_*`: First-order theory of mind questions (what an agent knows/believes)
- `second_order_*`: Second-order theory of mind questions (what an agent thinks another agent knows/believes)

## Setup

1. Clone this repository
2. Install the required packages:

```bash
pip install vllm tqdm
```

## Usage

### Step 1: Process the Data

First, run the data processing script to convert the raw data files into JSON format:

```bash
python process_data.py
```

This will create a `processed_data` directory containing JSON files for each dataset split.

### Step 2: Evaluate an LLM

To evaluate a language model using vLLM, run:

```bash
python evaluate_llm.py --model MODEL_NAME [--dataset DATASET] [--batch_size BATCH_SIZE] [--max_samples MAX_SAMPLES]
```

Arguments:
- `--model`: HuggingFace model name or path (required)
- `--dataset`: Dataset to evaluate on (train, val, or test; default: val)
- `--batch_size`: Batch size for inference (default: 8)
- `--max_samples`: Maximum number of samples to evaluate (optional)
- `--output_dir`: Directory to save evaluation results (default: evaluation_results)
- `--strict_matching`: Use strict exact matching for evaluation (optional)
- `--exclude_memory_reality`: Exclude memory and reality question types from evaluation

Example:
```bash
python evaluate_llm.py --model meta-llama/Llama-2-7b-hf --dataset val --batch_size 16 --max_samples 100 --exclude_memory_reality
```

### Testing a Single Example

You can also test a single example to see how the model performs:

```bash
python test_example.py --model MODEL_NAME [--json_file FILE --example_id ID | --story STORY --question QUESTION]
```

Arguments:
- `--model`: HuggingFace model name or path (required)
- `--json_file`: Path to a processed JSON file containing examples
- `--example_id`: Index of the example to use from the JSON file (default: 0)
- `--story`: Story text to use (alternative to --json_file)
- `--question`: Question to ask (required if using --story)
- `--answer`: Reference answer for comparison (optional)
- `--exclude_memory_reality`: Skip memory and reality questions when loading from JSON

Examples:
```bash
# Test with an example from a processed JSON file
python test_example.py --model meta-llama/Llama-2-7b-hf --json_file processed_data/val.json --example_id 5 --exclude_memory_reality

# Test with a custom story and question
python test_example.py --model meta-llama/Llama-2-7b-hf --story "Oliver entered the room. The apple is in the basket. Oliver moved the apple to the drawer." --question "Where is the apple now?" --answer "drawer"
```

### Results

The evaluation script will output:
- Overall accuracy
- Accuracy by question type
- Match type distribution
- Detailed results saved in a JSON file

## Answer Comparison

The evaluation script uses several methods to compare the model's answer with the ground truth:

1. **Exact Match**: The model's answer exactly matches the ground truth (case-insensitive)
2. **Normalized Match**: After removing non-alphabetic characters and articles, the answers match
3. **Contained Match**: The normalized ground truth is contained within the model's answer
4. **Prefix/Suffix Match**: The ground truth is at the beginning or end of the model's answer

This approach handles common variations in LLM outputs, such as:
- Additional explanations (e.g., "The answer is bucket")
- Articles or filler words (e.g., "the bucket" vs "bucket")
- Different formatting or punctuation

## Implementation Details

- **Token Limit**: The LLM's generation is limited to 10 tokens to ensure concise answers
- **Text Normalization**: Only alphabetic characters (a-z) are preserved; punctuation, numbers, and special characters are removed
- **Article Removal**: Articles ("a", "an", "the") are removed as whole words, but 'a' is preserved when it's part of another word
- **Question Filtering**: Memory and reality question types can be excluded from evaluation to focus on theory of mind reasoning

## Example Output

```
Processing val data...
Loaded 800 samples from processed_data/val.json
Filtered data to 480 examples after removing specified question types
Loading model: meta-llama/Llama-2-7b-hf
Evaluating 100 examples...
[...]

Evaluation complete. Results saved to evaluation_results/meta-llama-Llama-2-7b-hf_val.json
Overall accuracy: 0.7600

Accuracy by question type:
  first_order_0_no_tom: 0.8000
  second_order_0_no_tom: 0.6500
  first_order_1_tom: 0.7000
  second_order_1_no_tom: 0.6000

Match types distribution:
  exact_match: 0.5200
  normalized_match: 0.1400
  contained_match: 0.0600
  prefix_match: 0.0400
  suffix_match: 0.0000
  no_match: 0.2400
```

## Customization

- To modify how prompts are formatted, edit the `format_prompt` function in `evaluate_llm.py`
- To modify the answer comparison logic, edit the `check_answer_match` function

## References

If you find this code useful for your research, please cite the following paper in your publication:


```bibtex
@inproceedings{le-etal-2019-revisiting,
    title = "Revisiting the Evaluation of Theory of Mind through Question Answering",
    author = "Le, Matthew  and
      Boureau, Y-Lan  and
      Nickel, Maximilian",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1598",
    doi = "10.18653/v1/D19-1598",
    pages = "5872--5877"
}
```

## License 

This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

![](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)