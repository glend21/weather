# weather

## 1. rain
Extrapolate rain radar images to estimate short-term projections

General status notes (to my future self):

- the TVL1 algo with default params is there but is running excessively long
- as above, but then it crashes -- investigate
- the Farneback algo with fixed params produces a SSIM score of ~ 64% - 74% (very limited testing)
- interestingly, the similarity score steadily decreases along an image set. I don't think I'm doing
    anything silly, like comparing all generated images to the first. This could be an interesting
    bug to track down.

- need a method of determing the beest parameter set of n dimensions, without just circling the drain.
    the brute-force approach is too expensive (and ugly)

- visually, the generated images using Farneback (standard caveat -- limited testing) are almost usable

## 2. planner
Hit free public weather APIs to predict favourable conditions for photography. The conditions can be
chosen based on cloud, wind, rain, moon position, etc

Eventually this will be running as an AWS lambda or similar.
