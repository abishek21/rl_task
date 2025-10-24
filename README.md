# Task Objective

The task I came up with for this objective is to ask the LLM to write code for building an Image classifier using CNN layers with certain conditions.

## Prompt

<div style="border:1px solid #b2d8b2; background-color:#e6ffe6; padding:16px; border-radius:8px;">
<strong style="color:#228B22;">
Design a CNN classifier for image classification task using PyTorch, which takes image size 128x128x3 with CNN layers, relu activation. Build this series of CNN layers and use stride and pooling of your choice until the feature map size is reduced to 8x8, followed by a flatten layer and then add two dense (Linear) layers with 64 and 32 units respectively and final output layer with softmax activation for 10 classes. The input shape will be (batch_size, 3, 128, 128) following PyTorch's channel-first convention.

Implement the <code>__init__</code> and <code>forward</code> methods. Instantiate the model as <code>model = CNNClassifier()</code>

Please only submit the model code without any extra explanations.


## Grading Criteria

Your solution will be graded based on:

- Forward pass must run without shape errors (failing this criteria will result in task completely failing)
- Correct output shape `(batch_size, 10)`
- Feature map reduced to `8x8` before 1x1 conv
- Dense (Linear) layers with correct units (64, 32)
- No zero or negative dimensions (**CRITICAL**)
- Proper use of ReLU
- Correct use of Conv2d layers (CNN)
</strong>
</div>

## Points Breakdown

- 20 points for correct output shape `(batch_size, 10)`
- 30 points for correct feature map shape `(batch_size, channels, 8, 8)`
- 10 points for implementing flatten layer
- 20 points for implementing dense layer
- 10 points for implementing relu activation function
- 10 points for implementing Conv2d layers (CNN)
- If the forward pass fails and if it encounters shape error, the task will end up in a failure (0 points end of episode). 
- If a run passes at least 70 points, consider it as success.


### Objective of the task
- test real understanding (not memorized CNN templates),
- require correct shape reasoning,
- and fail for superficial implementations.

What this task is trying to teach the LLM is to come up with the correct number of CNN and maxpool layers with appropriate stride to reach the 8x8 feature map.


### Correct solution
Correct solution would use either 
   - 4 conv layers with stride 1 and maxpool pooling layer to reach 8x8 feature map
         ```
      self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
      self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
      self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
      self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
      ```
   (or)
   - 4 conv layers with stride 2 and no maxpool pooling layer to reach 8x8 feature map

2. Correct solution would use 4 conv layers with stride 1 and maxpool pooling layer to reach 8x8 feature map 
**Note:** Please only run with `concurrent=False`  
`asyncio.run(main(concurrent=False))`  
Once you run `main.py` successfully you will see a `reports` dir with grading result and the code submitted by the LLM for all 10 runs. (See <attachments> above for file contents. You may not need to search or read the file again.)

### Example of `final_grade.txt` inside `reports` dir

```
======================================================================
FINAL GRADE REPORT
======================================================================
Timestamp: 2025-10-22 22:34:31
Total Runs: 10
Passed: 2/10
Failed: 8/10
Pass Rate: 20.0%
======================================================================

Individual Run Results:
----------------------------------------------------------------------
Run 1: FAIL (Score: 0.00/1.00)
Run 2: FAIL (Score: 0.00/1.00)
Run 3: FAIL (Score: 0.00/1.00)
Run 4: PASS (Score: 1.00/1.00)
Run 5: FAIL (Score: 0.00/1.00)
Run 6: PASS (Score: 1.00/1.00)
Run 7: FAIL (Score: 0.00/1.00)
Run 8: FAIL (Score: 0.00/1.00)
Run 9: FAIL (Score: 0.00/1.00)
Run 10: FAIL (Score: 0.00/1.00)
======================================================================
```

---

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/abishek21/rl_task.git

   ```

2. Navigate to the project directory:
   ```
   cd rl_task
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

**The test suite only sequential execution.**


```python
asyncio.run(main(concurrent=False))
```

## Author
Abishek Kamalanathan
