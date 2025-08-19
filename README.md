This repository contains my solution for FNC-1 (Fake News Challenge: http://www.fakenewschallenge.org/).
The code and report in this repository were submitted as part of the 4th year Natural Language Processing Module to the Computer Science Department at Durham University. It was awarded a mark of 88% (2nd highest in the year).

# Dataset
The FNC dataset consists of 49,973 pairs of headlines and article bodies. The body text is annotated by the following classes: Agree, Disagree, Discuss, or Unrelated to the headline.

# Task
Input: A headline and a body text
Output: Classify the stance of the body text relative to the claim made in the headline into one of
four categories:  
    - Agrees: The body text agrees with the headline.  
    - Disagrees: The body text disagrees with the headline.  
    - Discusses: The body text discusses the same topic as the headline, but does not take a position.  
    - Unrelated: The body text discusses a different topic than the headline.  

# Repository Structure
report.pdf outlines the architecture and performance of two different natural language processing models, one more traditional classifier, and another generative "prompt-based" classifier. Comparison is made between the performance of both models in this task.

DeBERTa_classifier.ipynb contains the code for all data processing and visualisation in report.pdf, as well as the DeBERTA-based classifier. FLAN_T5_prompting_classifier.ipynb contains the code for the "prompt-based" FLAN-T5 classifier.

All training and validation data, including plots for each model in my report, can be found in modelDirectory. I removed the actual model weights due to their file size.