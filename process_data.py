import json
import os
import re

def process_file(txt_file, trace_file):
    """
    Process a pair of txt and trace files to extract stories, questions, answers, and question types.
    
    Args:
        txt_file (str): Path to the text file containing stories, questions, and answers
        trace_file (str): Path to the trace file containing question types
    
    Returns:
        list: List of dictionaries, each containing a data point
    """
    # Read files
    with open(txt_file, 'r', encoding='utf-8') as f:
        txt_content = f.readlines()
    
    with open(trace_file, 'r', encoding='utf-8') as f:
        trace_content = f.readlines()
    
    # Initialize variables
    data_points = []
    current_story = []
    current_data = {}
    line_num = 0
    
    # Process text file
    while line_num < len(txt_content):
        line = txt_content[line_num].strip()
        
        # Check if line starts with a number followed by a space
        if re.match(r'^\d+\s', line):
            # Extract the number and content
            parts = line.split(' ', 1)
            number = int(parts[0])
            
            # If number is 1 and we already have a story, we're starting a new data point
            if number == 1 and current_story and 'question' in current_data:
                data_points.append(current_data)
                current_data = {}
                current_story = []
            
            # Check if this line contains a question (has a question mark)
            if '?' in line:
                # This is a question line
                q_parts = parts[1].split('?', 1)
                question = q_parts[0] + '?'
                # Extract answer, removing the number at the end
                answer_parts = q_parts[1].strip().split('\t')
                answer = answer_parts[0].strip()
                
                current_data['question'] = question
                current_data['answer'] = answer
                current_data['story'] = ' '.join(current_story)
            else:
                # This is a story line
                current_story.append(parts[1])
        
        line_num += 1
    
    # Add the last data point if it exists
    if current_story and 'question' in current_data:
        data_points.append(current_data)
    
    # Now add question types from trace file
    for i, trace_line in enumerate(trace_content):
        if i < len(data_points):
            parts = trace_line.strip().split(',')
            # The question type is the second-to-last element
            question_type = parts[-2] if len(parts) >= 2 else "unknown"
            data_points[i]['question_type'] = question_type
    
    return data_points

def process_all_files():
    """Process all data files and save as JSON"""
    data_dir = 'data'
    output_dir = 'processed_data'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each set (train, val, test)
    for dataset in ['train', 'val', 'test']:
        txt_file = os.path.join(data_dir, f'{dataset}.txt')
        trace_file = os.path.join(data_dir, f'{dataset}.trace')
        
        if os.path.exists(txt_file) and os.path.exists(trace_file):
            print(f"Processing {dataset} data...")
            data_points = process_file(txt_file, trace_file)
            
            # Save to JSON file
            output_file = os.path.join(output_dir, f'{dataset}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_points, f, indent=2)
            
            print(f"Saved {len(data_points)} data points to {output_file}")
        else:
            print(f"Missing files for {dataset}")

if __name__ == "__main__":
    process_all_files() 