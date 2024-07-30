# Condensation tracker

This repository contains a Python script for tracking objects in a video using a particle filter-based method known as the [Condensation algorithm](http://vision.stanford.edu/teaching/cs231b_spring1415/papers/isard-blake-98.pdf). The tracker estimates the position of a bounding box over time by propagating particles, observing weights, and resampling based on observed data. It also allows for the visualization of the tracking process.

## Features
- **Object Tracking:** Track objects across frames using a particle filter.
- **Histogram-based Observation:** Utilizes color histograms to observe and update the particle weights.
- **Interactive Bounding Box Selection:** Allows users to select the initial bounding box for tracking.
- **Visualization:** Provides a visual representation of the tracking process.

## How to use it
- Install all the requirements from requirements.txt (I am using Python 3.8.18)
- Run 
```
python code/condensation_tracker.py
```
- Choose a starting bounding box for the object to be tracked and then press 'q'
- Press 'q' to exit the window

## Contribution
Contributions are welcome! Feel free to contribute by implementing new features, improvements, or bug fixes . Please open an issue or pull request on GitHub.
