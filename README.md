hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
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

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

## Solution:

**Author: Abishek Kamalanathan**

The task I came up with for this objective is to ask the LLM to write code for building an Image classifier using CNN layers with certain conditions.

### Prompt:
```
Design a CNN classifier for image classification task using PyTorch, which takes image size 128x128x3 with CNN layers, relu activation and max pooling layers. Build this series of CNN layers and use stride of your choice until the feature map size is reduced to 8x8 , followed by a flatten layer and then add two dense (Linear) layers with 64 and 32 units respectively and final output layer with softmax activation for 10 classes.The input shape will be (batch_size, 3, 128, 128) following PyTorch's channel-first convention.
Implement the __init__ and forward methods. Instantiate the model as 'model = CNNClassifier()'
Please only submit the model code without any extra explanations.
Your solution will be graded based on:
Forward pass must run without shape errors failing this criteria will result in task completely failing
Correct output shape (batch_size, 10) 
Feature map reduced to 8x8 before 1x1 conv
Dense (Linear) layers with correct units (64, 32)
No zero or negative dimensions (CRITICAL)
Proper use of ReLU and MaxPool
```

### My Grading logic:
- 20 points for correct output shape (batch_size,10)
- 25 points for correct feature map shape (batch_size,channels,8,8)
- 10 points for implementing flatten layer
- 20 points for implementing dense layer
- 10 points for implementing relu activation function
- 10 points for implementing maxpool layer
- 10 points for implementing Conv2d layers(CNN)

If the forward pass fails and if it encounters shape error, the task will end up in a failure (0 points end of episode)

If a run passes at least 70 points, consider it as success

### What this task is trying to teach the LLM:
What this task is trying to teach the LLM is to come up with the correct number of CNN and maxpool layers with appropriate stride to reach the 8x8 feature map.

### Note:
Please only run with concurrent=False:
```python
asyncio.run(main(concurrent=False))
```

Once you run main.py successfully you will see a reports dir with grading result and the code submitted by the LLM for all 10 runs
