# Final Project Log CART 398/IMCA 221

This will serve as a form of documentation to help itterate on the final project. The repository currently has only a python file with the osc protocol with kinect serving as the input. Will be adding both max patches as well. 

12/11/25

**Configuration Errors**

Problem: "Input tap too large" and "Wrong Point Size" errors when adding depth as third input

What happened:

Model wasn't configured for 3 inputs
Buffer only had 2 sample slots instead of 3
Datasets missing dimension specifications

Solution:

* fluid.mlpregressor~ @numins 3 @numouts 10
* buffer~ xybuf @samps 3
* fluid.dataset~ xydata @dims 3
* fluid.dataset~ paramsdata @dims 10

**Poor Training Performance** 

Problem: Model stuck at high error (0.3+), couldn't fit properly
What happened:

* Too few iterations (10000)
* Hidden layers too small (3 neurons)
* Mismatched input scaling (x,y at 0-255, depth at 120-255, outputs at 0-1)

Solution:

* Increased @maxiter to 100000-200000
* Normalize ALL inputs to 0-1 range:

    * X: expr $f1 / 127
    * Y: expr $f1 / 127
    * Depth: expr ($f1 - 120) / 135
* Increase @hiddenlayers to 64 32 16 or larger



Increase @hiddenlayers to 64 32 16 or larger

**Depth Not Affecting Synth Output**


What happened:

* Depth values in training data likely not varying enough
* Depth scaled differently than x/y, giving it less influence

Solution:

* Verify depth variation: print dataset, check third column shows wide range (e.g., 0.3-0.9, not 0.71-0.74) THIS DID NOT WORK
* Train with examples at significantly different depths (move closer/farther from Kinect)
* Ensure synth parameters actually change with depth in training examples
* Use consistent 0-1 normalization for all inputsProblem: Model semi-working but depth changes don't influence synth, only x/y plane does


