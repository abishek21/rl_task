import os
import asyncio
import json
import io
import re
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict
from datetime import datetime

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to output something; stdout will be captured and returned.
    """
    try:
        # Provide the necessary imports for PyTorch
        namespace = {
            "torch": torch,
            "nn": nn,
            "F": F,
            "np": np,
        }
        
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


def grade_cnn_classifier(model_code: str):
    """
    Grades a submitted CNN classifier model (PyTorch).
    
    Grading criteria:
    1. Model accepts input shape (batch_size, 3, 128, 128)
    2. Output shape is (batch_size, 10) for 10 classes
    3. Feature map is reduced to 8x8
    4. Has flatten layer
    5. Has two dense layers with 64 and 32 units
    6. Final layer has 10 units with softmax
    7. Uses ReLU activation and max pooling
    """
    env = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
    }
    
    try:
        exec(model_code, env)
    except Exception as e:
        return {"score": 0.0, "reason": f"Code failed to exec: {e}", "success": False}

    # Look for CNNClassifier class
    if 'CNNClassifier' not in env:
        return {"score": 0.0, "reason": "No class named 'CNNClassifier' found in submission", "success": False}
    
    obj = env['CNNClassifier']
    
    # Check if it's a class and subclass of nn.Module
    if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
        return {"score": 0.0, "reason": "CNNClassifier is not a valid nn.Module class", "success": False}
    
    # Try to instantiate the model
    try:
        model = obj()
    except:
        try:
            model = obj(10)  # Try with num_classes argument
        except Exception as e:
            return {"score": 0.0, "reason": f"Failed to instantiate CNNClassifier: {e}", "success": False}

    model.eval()
    
    # Test with a sample input
    batch_size = 2
    try:
        test_input = torch.randn(batch_size, 3, 128, 128)
        
        # Hook to capture intermediate shapes
        shapes = []
        hooks = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                shapes.append({
                    "layer": module.__class__.__name__,
                    "output_shape": tuple(output.shape)
                })
        
        # Register hooks on all layers
        for layer in model.modules():
            if layer != model:  # Skip the model itself
                hooks.append(layer.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
    except Exception as e:
        return {
            "score": 0.0,
            "reason": f"Forward pass failed (likely due to feature map shape mismatch): {e}",
            "success": False,
            "shapes": shapes if shapes else None
        }
    
    # Grading criteria
    score = 0.0
    max_score = 0.0
    details = {}
    issues = []
    # 1. Check output shape (20 points)
    max_score += 20
    if output.shape == (batch_size, 10):
        score += 20
        details["output_shape"] = "✓ Correct (batch_size, 10)"
    else:
        issues.append(f"Output shape is {tuple(output.shape)}, expected ({batch_size}, 10)")
        details["output_shape"] = f"✗ Wrong: {tuple(output.shape)}"

    # 2. Check for 8x8 feature map (25 points)
    max_score += 25
    found_8x8 = False
    
    for i, shape_info in enumerate(shapes):
        shape = shape_info["output_shape"]
        # Check if we have a 8x8 spatial dimension (shape would be [batch, channels, 8, 8])
        if len(shape) == 4 and shape[2] == 8 and shape[3] == 8:
            found_8x8 = True
            break

    if found_8x8:
        score += 25
        details["feature_map_8x8"] = "✓ Found 8x8 feature map"
    else:
        issues.append("Feature map not reduced to 8x8")
        details["feature_map_8x8"] = "✗ Not reduced to 8x8"

    # 3. Check for flatten operation (10 points)
    max_score += 10
    found_flatten = False
    for i, shape_info in enumerate(shapes):
        if "Flatten" in shape_info["layer"]:
            found_flatten = True
            break
        # Check if we went from 4D to 2D
        if i > 0:
            prev_shape = shapes[i-1]["output_shape"]
            curr_shape = shape_info["output_shape"]
            if len(prev_shape) == 4 and len(curr_shape) == 2:
                found_flatten = True
                break
    
    if found_flatten:
        score += 15
        details["flatten"] = "✓ Found flatten layer"
    else:
        issues.append("No flatten layer found")
        details["flatten"] = "✗ Missing flatten layer"
    
    # 4. Check for dense layers with correct units (20 points)
    max_score += 20
    found_dense_64 = False
    found_dense_32 = False
    
    # Look for Linear layers in the architecture
    for i, shape_info in enumerate(shapes):
        if "Linear" in shape_info["layer"]:
            shape = shape_info["output_shape"]
            if len(shape) == 2:
                units = shape[1]
                if units == 64:
                    found_dense_64 = True
                elif units == 32:
                    found_dense_32 = True
    
    if found_dense_64:
        score += 10
        details["dense_64"] = "✓ Found dense layer with 64 units"
    else:
        issues.append("Missing dense layer with 64 units")
        details["dense_64"] = "✗ Missing dense layer with 64 units"
    
    if found_dense_32:
        score += 10
        details["dense_32"] = "✓ Found dense layer with 32 units"
    else:
        issues.append("Missing dense layer with 32 units")
        details["dense_32"] = "✗ Missing dense layer with 32 units"
    
    # 5. Check for ReLU activation (check both nn.ReLU layers and F.relu in code) (10 points)
    max_score += 10
    has_relu_layer = any("ReLU" in shape_info["layer"] for shape_info in shapes)
    has_relu_functional = "F.relu" in model_code or "F.ReLU" in model_code or "torch.relu" in model_code
    has_relu = has_relu_layer or has_relu_functional
    
    if has_relu:
        score += 10
        if has_relu_functional:
            details["relu"] = "✓ Uses F.relu activation"
        else:
            details["relu"] = "✓ Uses ReLU activation"
    else:
        issues.append("No ReLU activation found")
        details["relu"] = "✗ Missing ReLU"
    
    # 6. Check for MaxPool layers (10 points)
    max_score += 10
    has_maxpool = any("MaxPool" in shape_info["layer"] for shape_info in shapes)
    
    if has_maxpool:
        score += 10
        details["maxpool"] = "✓ Uses MaxPool layers"
    else:
        issues.append("No MaxPool layers found")
        details["maxpool"] = "✗ Missing MaxPool"
    
    # 7. Check architecture contains Conv2d layers (10 points)
    max_score += 10
    num_conv_layers = sum(1 for shape_info in shapes if "Conv2d" in shape_info["layer"])
    if num_conv_layers >= 3:
        score += 10
        details["conv_layers"] = f"✓ Found {num_conv_layers} Conv2d layers"
    elif num_conv_layers > 0:
        score += 5
        details["conv_layers"] = f"⚠ Only {num_conv_layers} Conv2d layers"
        issues.append(f"Expected at least 3 Conv2d layers, found {num_conv_layers}")
    else:
        details["conv_layers"] = "✗ No Conv2d layers found"
        issues.append("No Conv2d layers found")
    
    # Get number of parameters for reporting
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Normalize score to 0-1
    normalized_score = score / max_score if max_score > 0 else 0.0
    success = normalized_score >= 0.7
    
    return {
        "score": round(normalized_score, 2),
        "raw_score": f"{score:.1f}/{max_score}",
        "success": success,
        "details": details,
        "issues": issues,
        "output_shape": tuple(output.shape),
        "intermediate_shapes": shapes,
        "num_parameters": num_params if num_params > 0 else 0,
        "full_code": model_code
    }


def save_report(run_id: int, success: bool, final_code: str, grade_result: dict, report_dir: str = "reports"):
    """Save a detailed report for each run."""
    # Create reports directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Determine status for filename
    status = "success" if success else "fail"
    filename = f"report_run_{run_id}_{status}.txt"
    filepath = os.path.join(report_dir, filename)
    
    # Generate report content
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"RUN {run_id} REPORT - {status.upper()}")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Grading results
    report_lines.append("GRADING RESULTS:")
    report_lines.append("-" * 70)
    report_lines.append(f"Overall Score: {grade_result.get('score', 0):.2f}/1.00")
    report_lines.append(f"Raw Score: {grade_result.get('raw_score', 'N/A')}")
    report_lines.append(f"Status: {'✓ PASS' if success else '✗ FAIL'}")
    report_lines.append("")
    
    if grade_result.get("output_shape"):
        report_lines.append(f"Final Output Shape: {grade_result['output_shape']}")
    
    if grade_result.get("num_parameters"):
        report_lines.append(f"Total Parameters: {grade_result['num_parameters']:,}")
    
    report_lines.append("")
    
    # Grading details
    if grade_result.get("details"):
        report_lines.append("Grading Details:")
        report_lines.append("-" * 70)
        for key, value in grade_result["details"].items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
    
    # Issues
    if grade_result.get("issues"):
        report_lines.append("Issues Found:")
        report_lines.append("-" * 70)
        for issue in grade_result["issues"]:
            report_lines.append(f"  • {issue}")
        report_lines.append("")
    
    # Reason for failure
    if grade_result.get("reason"):
        report_lines.append(f"Failure Reason: {grade_result['reason']}")
        report_lines.append("")
    
    # Final submitted code
    report_lines.append("=" * 70)
    report_lines.append("FINAL SUBMITTED CODE:")
    report_lines.append("=" * 70)
    if final_code:
        report_lines.append(final_code)
    else:
        report_lines.append("No code was submitted")
    report_lines.append("")
    report_lines.append("=" * 70)
    
    # Write report to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return filepath


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=4096, tools=tools, messages=messages
        )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print("```python")
                            for line in tool_input["expression"].split("\n"):
                                print(f"{line}")
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result.get("result", "") if result.get("result") else result.get("error", ""))
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use in response, ending loop
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_cnn_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    verbose: bool = False,
    report_dir: str = "reports",
) -> tuple[int, bool, Any]:
    """Run a single test for the CNN classifier task."""
    
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    # Track only the final submitted code
    final_code = None
    
    # Custom handler to capture only submitted answer
    original_submit = tool_handlers["submit_answer"]
    
    def submit_answer_wrapper(answer: Any) -> SubmitAnswerToolResult:
        nonlocal final_code
        final_code = answer
        return original_submit(answer)
    
    # Use the wrapper for this test
    modified_handlers = tool_handlers.copy()
    modified_handlers["submit_answer"] = submit_answer_wrapper

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=modified_handlers,
        max_steps=10,
        verbose=verbose,
    )

    # Use the final submitted code for grading
    code_to_grade = final_code if final_code else ""
    
    if verbose:
        print("\n" + "="*60)
        print("GRADING THE SUBMISSION")
        print("="*60)
    
    grade_result = grade_cnn_classifier(code_to_grade)
    
    success = grade_result.get("success", False)
    
    # Save report to file
    report_path = save_report(run_id, success, final_code or "", grade_result, report_dir)
    print(f"\nReport saved to: {report_path}")
    
    # Print detailed grading results
    print(f"\n{'='*50}")
    print(f"Run {run_id} Grading Results:")
    print(f"{'='*50}")
    print(f"Overall Score: {grade_result['score']:.2f}/1.00")
    print(f"Raw Score: {grade_result.get('raw_score', 'N/A')}")
    print(f"Status: {'✓ PASS' if success else '✗ FAIL'}")
    
    if grade_result.get("output_shape"):
        print(f"\nFinal Output Shape: {grade_result['output_shape']}")
    
    if grade_result.get("num_parameters"):
        print(f"Total Parameters: {grade_result['num_parameters']:,}")
    
    # Print grading details
    if grade_result.get("details"):
        print(f"\nGrading Details:")
        for key, value in grade_result["details"].items():
            print(f"  {key}: {value}")
    
    # Print issues if any
    if grade_result.get("issues"):
        print(f"\nIssues Found:")
        for issue in grade_result["issues"]:
            print(f"  • {issue}")
    
    # Print intermediate shapes for debugging
    if grade_result.get("intermediate_shapes") and verbose:
        print(f"\nIntermediate Shapes:")
        for i, shape_info in enumerate(grade_result["intermediate_shapes"]):
            print(f"  {i+1}. {shape_info['layer']}: {shape_info['output_shape']}")
    
    if grade_result.get("reason"):
        print(f"\nReason: {grade_result['reason']}")
    
    # Print only the final submitted code
    print("\n" + "="*50)
    print("FINAL SUBMITTED CODE:")
    print("="*50)
    if final_code:
        print(final_code)
    else:
        print("No code was submitted")
    print("="*50)

    return run_id, success, grade_result


async def main(concurrent: bool = False):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to execute. torch, nn (torch.nn), and F (torch.nn.functional) are already imported.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "Final CNN classifier implementation"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    num_runs = 10
    report_dir = "reports"
    
    # Create reports directory
    os.makedirs(report_dir, exist_ok=True)
    
    prompt = """Design a CNN classifier for image classification task using PyTorch, which takes image size 128x128x3 with 
CNN layers, with relu activation and max pooling layers. Build this series of CNN layers and use stride of your choice until the feature map size is reduced to 8x8.
followed by flatten the output and add two dense (Linear) layers with 64 and 32 units respectively and final output layer with softmax activation for 10 classes.

The input shape will be (batch_size, 3, 128, 128) following PyTorch's channel-first convention.

Implement the __init__ and forward methods. Instantiate the model as 'model = CNNClassifier()'

Please only submit the model code without any extra explanations.

Your solution will be graded based on:
- Forward pass must run without shape errors failing this criteria will result in task completely failing
- Correct output shape (batch_size, 10) 
- Feature map reduced to 8x8 before 1x1 conv
- Dense (Linear) layers with correct units (64, 32)
- No zero or negative dimensions (CRITICAL)
- Proper use of ReLU and MaxPool


.
"""

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iteration(s) {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_cnn_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            verbose=True,  # Set to True to see the agent's work
            report_dir=report_dir,
        )
        for i in range(num_runs)
    ]

    # Run tests
    if concurrent:
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(1 for _, success, _ in results if success)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")
    
    # Save final grade report
    final_grade_path = os.path.join(report_dir, "final_grade.txt")
    with open(final_grade_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL GRADE REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runs: {num_runs}\n")
        f.write(f"Passed: {successes}/{num_runs}\n")
        f.write(f"Failed: {num_runs - successes}/{num_runs}\n")
        f.write(f"Pass Rate: {pass_rate:.1f}%\n")
        f.write("=" * 70 + "\n")
        f.write("\nIndividual Run Results:\n")
        f.write("-" * 70 + "\n")
        for run_id, success, grade_result in results:
            status = "PASS" if success else "FAIL"
            score = grade_result.get('score', 0)
            f.write(f"Run {run_id}: {status} (Score: {score:.2f}/1.00)\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nFinal grade report saved to: {final_grade_path}")


if __name__ == "__main__":
    asyncio.run(main(concurrent=False))


