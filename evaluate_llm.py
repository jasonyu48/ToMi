import json
import os
import re
import string
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams

def load_data(file_path):
    """Load processed data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def format_prompt(data_point):
    """Format a single data point into a prompt for the model"""
    story = data_point['story']
    question = data_point['question']
    return f"Read the following story and answer the question with a single word or phrase.\n\nStory: {story}\n\nQuestion: {question}\n\nAnswer:"

def normalize_text(text):
    """Normalize text for better comparison"""
    # Convert to lowercase
    text = text.lower()
    # Only keep a-z characters
    text = re.sub(r'[^a-z]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove articles as whole words (not part of words)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    return text.strip()

def check_answer_match(prediction, ground_truth):
    """
    Check if the prediction matches the ground truth, with various normalization techniques
    
    Args:
        prediction (str): The model's prediction
        ground_truth (str): The ground truth answer
        
    Returns:
        tuple: (is_correct, match_type) where match_type describes how the match was determined
    """
    # Exact match (already lowercase)
    if prediction == ground_truth:
        return True, "exact_match"
    
    # Normalized match
    norm_pred = normalize_text(prediction)
    norm_truth = normalize_text(ground_truth)
    
    if norm_pred == norm_truth:
        return True, "normalized_match"
    
    # Check if prediction contains the ground truth (for cases where LLM adds explanations)
    if norm_truth in norm_pred:
        return True, "contained_match"
    
    # Check if ground truth is at the start of the prediction
    if norm_pred.startswith(norm_truth):
        return True, "prefix_match"
    
    # Check if ground truth is at the end of the prediction
    if norm_pred.endswith(norm_truth):
        return True, "suffix_match"
        
    return False, "no_match"

def evaluate_model(model_name, data, batch_size=8, max_samples=None, filter_question_types=None):
    """
    Evaluate a model on the given data.
    
    Args:
        model_name (str): HuggingFace model name or path
        data (list): List of data points
        batch_size (int): Batch size for inference
        max_samples (int, optional): Maximum number of samples to evaluate
        filter_question_types (list, optional): List of question types to exclude
    
    Returns:
        dict: Dictionary with evaluation results
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
    
    # Filter out memory and reality questions if requested
    if filter_question_types:
        data = [dp for dp in data if dp['question_type'] not in filter_question_types]
        print(f"Filtered data to {len(data)} examples after removing specified question types")
    
    if max_samples is not None:
        data = data[:max_samples]
    
    # Prepare prompts
    prompts = [format_prompt(dp) for dp in data]
    
    # Run inference in batches
    print(f"Evaluating {len(prompts)} examples...")
    all_outputs = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)
    
    # Process results
    results = []
    correct = 0
    match_types = {"exact_match": 0, "normalized_match": 0, "contained_match": 0, 
                  "prefix_match": 0, "suffix_match": 0, "no_match": 0}
    
    for i, output in enumerate(all_outputs):
        generated_text = output.outputs[0].text.strip().lower()
        gold_answer = data[i]['answer'].lower()
        
        # Check if the answer is correct with improved matching
        is_correct, match_type = check_answer_match(generated_text, gold_answer)
        
        if is_correct:
            correct += 1
        
        # Update match type counter
        match_types[match_type] += 1
        
        results.append({
            'story': data[i]['story'],
            'question': data[i]['question'],
            'gold_answer': gold_answer,
            'predicted_answer': generated_text,
            'question_type': data[i]['question_type'],
            'correct': is_correct,
            'match_type': match_type
        })
    
    # Calculate accuracy
    accuracy = correct / len(results) if results else 0
    
    # Calculate accuracy by question type
    type_accuracy = {}
    type_counts = {}
    
    for result in results:
        q_type = result['question_type']
        if q_type not in type_accuracy:
            type_accuracy[q_type] = 0
            type_counts[q_type] = 0
        
        type_counts[q_type] += 1
        if result['correct']:
            type_accuracy[q_type] += 1
    
    for q_type in type_accuracy:
        type_accuracy[q_type] = type_accuracy[q_type] / type_counts[q_type]
    
    return {
        'total_accuracy': accuracy,
        'accuracy_by_type': type_accuracy,
        'match_types': {k: v / len(results) for k, v in match_types.items()},
        'results': results
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on ToM reasoning tasks")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, default="val", choices=["train", "val", "test"], 
                        help="Dataset to evaluate on (train, val, or test)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--strict_matching", action="store_true", 
                        help="Use strict exact matching for evaluation")
    parser.add_argument("--exclude_memory_reality", action="store_true",
                        help="Exclude memory and reality question types")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_file = f"processed_data/{args.dataset}.json"
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Please run process_data.py first.")
        return
    
    data = load_data(data_file)
    print(f"Loaded {len(data)} samples from {data_file}")
    
    # Set filter for question types
    filter_question_types = ["memory", "reality"] if args.exclude_memory_reality else None
    
    # Evaluate model
    results = evaluate_model(
        args.model,
        data,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        filter_question_types=filter_question_types
    )
    
    # Save results
    output_file = os.path.join(
        args.output_dir, 
        f"{args.model.replace('/', '-')}_{args.dataset}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nEvaluation complete. Results saved to {output_file}")
    print(f"Overall accuracy: {results['total_accuracy']:.4f}")
    print("\nAccuracy by question type:")
    for q_type, acc in results['accuracy_by_type'].items():
        print(f"  {q_type}: {acc:.4f}")
    
    print("\nMatch types distribution:")
    for match_type, percentage in results['match_types'].items():
        print(f"  {match_type}: {percentage:.4f}")

if __name__ == "__main__":
    main() 