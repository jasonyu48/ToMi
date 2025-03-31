import json
import argparse
from vllm import LLM, SamplingParams
from evaluate_llm import format_prompt, check_answer_match

def test_single_example(model_name, story, question, reference_answer=None):
    """
    Test a single example using vLLM.
    
    Args:
        model_name (str): HuggingFace model name or path
        story (str): The story text
        question (str): The question to ask
        reference_answer (str, optional): Reference answer for comparison
        
    Returns:
        str: The model's response
    """
    print(f"Loading model: {model_name}")
    
    # Initialize the model
    llm = LLM(model=model_name)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Use greedy decoding
        max_tokens=10,    # Limit to 10 tokens
        stop=["</s>", "\n"]
    )
    
    # Format the prompt
    data_point = {
        'story': story,
        'question': question
    }
    prompt = format_prompt(data_point)
    
    # Run inference
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    
    # Compare with reference answer if provided
    if reference_answer:
        is_correct, match_type = check_answer_match(generated_text.lower(), reference_answer.lower())
        print(f"\nReference answer: {reference_answer}")
        print(f"Model answer: {generated_text}")
        print(f"Correct: {is_correct} (Match type: {match_type})")
    
    return generated_text

def load_example_from_json(json_file, example_id, skip_memory_reality=False):
    """
    Load a specific example from a processed JSON file.
    
    Args:
        json_file (str): Path to the processed JSON file
        example_id (int): Index of the example to load
        skip_memory_reality (bool): Whether to skip memory and reality questions
        
    Returns:
        dict: The example data
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Filter out memory and reality questions if requested
    if skip_memory_reality:
        data = [dp for dp in data if dp['question_type'] not in ["memory", "reality"]]
        print(f"Filtered to {len(data)} examples after removing memory and reality questions")
    
    if example_id >= len(data):
        raise ValueError(f"Example ID {example_id} is out of range. File contains {len(data)} examples.")
    
    return data[example_id]

def main():
    parser = argparse.ArgumentParser(description="Test a single example with an LLM")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    
    # Input options group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--json_file", type=str, help="Path to processed JSON file with examples")
    input_group.add_argument("--story", type=str, help="Story text to use for testing")
    
    # Other arguments
    parser.add_argument("--example_id", type=int, default=0, help="Example ID to use from JSON file")
    parser.add_argument("--question", type=str, help="Question to ask (required if using --story)")
    parser.add_argument("--answer", type=str, help="Reference answer (optional)")
    parser.add_argument("--exclude_memory_reality", action="store_true",
                        help="Exclude memory and reality question types when loading from JSON")
    
    args = parser.parse_args()
    
    # Get story and question
    if args.json_file:
        example = load_example_from_json(args.json_file, args.example_id, args.exclude_memory_reality)
        story = example['story']
        question = example['question']
        reference_answer = example['answer']
        question_type = example['question_type']
        print(f"Using example {args.example_id} from {args.json_file}")
        print(f"Question type: {question_type}")
        print(f"Story: {story}")
        print(f"Question: {question}")
    else:
        if not args.question:
            parser.error("--question is required when using --story")
        story = args.story
        question = args.question
        reference_answer = args.answer
    
    # Run the model
    result = test_single_example(args.model, story, question, reference_answer)
    
    if not args.answer and not args.json_file:
        print(f"\nModel answer: {result}")

if __name__ == "__main__":
    main() 