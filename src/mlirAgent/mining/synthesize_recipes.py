import argparse
import json
import os

from tqdm import tqdm

# TODO: Get a proper model api hooked up here!

RECIPE_PROMPT = """
You are a Compiler Engineer writing a 'Cookbook' for an AI Agent.
Analyze this Git Commit which contains a BUG FIX or OPTIMIZATION.

CONTEXT:
- Commit Message: {msg}
- GitHub Labels: {labels}
- C++ Changes (Logic): 
{diffs}
- MLIR Tests (Verification): 
{tests}

TASK:
Create a structured YAML recipe.
If the labels include 'crash', focus on stability.
If 'missed-optimization', focus on performance.

OUTPUT FORMAT (YAML):
id: "git-{hash}"
tags: [{labels}, inferred_keywords]
problem:
  summary: "One sentence description"
  description: "Detailed context from commit msg"
solution:
  pattern: "Explain the coding pattern (e.g. 'Use rewriter.replaceOp')"
  changes:
    - file: "filename.cpp"
      action: "modify"
      explanation: "Why this change?"
verification:
  input_ir_pattern: "Snippet from test input"
  expected_output: "Snippet from CHECK lines"
"""

def synthesize(input_file: str, output_dir: str):
    print(f"🧪 Synthesizing Recipes from {input_file}...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file) as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="AI Processing"):
        data = json.loads(line)
        commit_hash = data['hash']
        
        output_path = os.path.join(output_dir, f"recipe_{commit_hash}.yaml")
        if os.path.exists(output_path):
            continue
            
        # Format Context
        diffs = "\n".join([f"File: {c['path']}\n{c['diff'][:2000]}" for c in data['changes']])
        tests = "\n".join([f"File: {t['path']}\n{t['content'][:1000]}" for t in data['tests']])
        
        labels = ", ".join(data.get('github_labels', []))
        
        prompt = RECIPE_PROMPT.format(
            msg=data['msg'],
            labels=labels, 
            diffs=diffs,
            tests=tests,
            hash=commit_hash
        )
        
        try:
            response = model.generate(prompt)
            yaml_content = response.replace("```yaml", "").replace("```", "").strip()
            
            with open(output_path, 'w') as out:
                out.write(yaml_content)
                
        except Exception as e:
            print(f"Failed {commit_hash}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    synthesize(args.input, args.output_dir)