import json
from pathlib import Path

out_path = Path("data/eval_questions_large.jsonl")

base_questions = [
    {"category": "math", "question": "What is the derivative of x^2?", "choices": ["x", "2x", "x^3", "2"], "answer": "B"},
    {"category": "math", "question": "What is the derivative of sin(x)?", "choices": ["cos(x)", "-cos(x)", "sin(x)", "-sin(x)"], "answer": "A"},
    {"category": "math", "question": "What is the integral of 2x dx?", "choices": ["x^2 + C", "2x + C", "x + C", "2x^2 + C"], "answer": "A"},
    {"category": "math", "question": "If a fair coin is flipped twice, what is the probability of two heads?", "choices": ["1/2", "1/3", "1/4", "3/4"], "answer": "C"},

    {"category": "ml", "question": "What is overfitting?", "choices": ["Performing well on training data but poorly on unseen data", "Using no parameters", "Training with no loss function", "Having too little training accuracy"], "answer": "A"},
    {"category": "ml", "question": "What is the purpose of a validation set?", "choices": ["To tune model choices before final testing", "To replace the optimizer", "To remove all features", "To guarantee zero test error"], "answer": "A"},
    {"category": "ml", "question": "What does regularization usually help reduce?", "choices": ["Overfitting", "Dataset size", "Input dimension only", "The number of classes"], "answer": "A"},
    {"category": "ml", "question": "In classification, what does cross-entropy commonly measure?", "choices": ["Mismatch between predicted probabilities and true labels", "The number of model layers", "The size of the dataset", "The learning rate schedule"], "answer": "A"},

    {"category": "reasoning", "question": "All dogs are mammals. Rex is a dog. What follows?", "choices": ["Rex is a mammal", "Rex is a bird", "All mammals are dogs", "No dogs are mammals"], "answer": "A"},
    {"category": "reasoning", "question": "If today is Monday, what day is two days after tomorrow?", "choices": ["Tuesday", "Wednesday", "Thursday", "Friday"], "answer": "C"},
    {"category": "reasoning", "question": "All squares are rectangles. Some rectangles are blue. What must be true?", "choices": ["All squares are blue", "Some blue objects are squares", "Squares are rectangles", "No rectangles are squares"], "answer": "C"},
    {"category": "reasoning", "question": "A is taller than B. B is taller than C. Who is tallest?", "choices": ["A", "B", "C", "Cannot determine"], "answer": "A"},

    {"category": "systems", "question": "What does caching usually reduce?", "choices": ["Repeated computation or access latency", "Correctness", "Source-code length only", "All memory usage"], "answer": "A"},
    {"category": "systems", "question": "What is batching commonly used for in ML inference?", "choices": ["Improving hardware utilization by processing multiple inputs together", "Deleting model weights", "Removing the need for memory", "Making every request slower"], "answer": "A"},
    {"category": "systems", "question": "What is a common bottleneck in transformer inference?", "choices": ["Attention computation and memory bandwidth", "The absence of labels", "Too many classes only", "CSV formatting"], "answer": "A"},
    {"category": "systems", "question": "What does a profiler help identify?", "choices": ["Runtime and resource bottlenecks", "Only syntax errors", "Only spelling mistakes", "The final test labels"], "answer": "A"},
]

expanded = []
for i in range(13):
    for q in base_questions:
        item = dict(q)
        item["question"] = f"{q['question']} Variant {i+1}."
        expanded.append(item)

with out_path.open("w", encoding="utf-8") as f:
    for item in expanded:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Wrote {len(expanded)} examples to {out_path}")
