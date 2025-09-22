# analyze-headers.py

import os
import re

def analyze_headers(directory):
    header_info = {}

    # Regular expressions to match class and function definitions
    class_pattern = re.compile(r'^\s*class\s+(\w+)')
    function_pattern = re.compile(r'^\s*[\w\s\*&]+(\w+)\s*\(.*\)\s*;?')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.hpp'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.readlines()
                    classes = []
                    functions = []
                    for line in content:
                        class_match = class_pattern.match(line)
                        function_match = function_pattern.match(line)
                        if class_match:
                            classes.append(class_match.group(1))
                        if function_match:
                            functions.append(function_match.group(1))
                    header_info[file_path] = {
                        'classes': classes,
                        'functions': functions
                    }
    return header_info

def main():
    directory = 'include/alpaka'  # Adjust this path as necessary
    header_analysis = analyze_headers(directory)

    # Output the analysis results
    for file_path, info in header_analysis.items():
        print(f'File: {file_path}')
        print(f'  Classes: {", ".join(info["classes"])}')
        print(f'  Functions: {", ".join(info["functions"])}')
        print()

if __name__ == '__main__':
    main()
