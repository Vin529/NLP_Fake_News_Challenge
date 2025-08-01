This reposity contains my solution for FNC-1 (Fake News Challenge: http://www.fakenewschallenge.org/).

report.pdf outlines the architecture and performance of two different natural language processing models, one traditional nlp classifier, and another "prompt based" classifier.
Comparison is made between the performance of both models in this task.

DeBERTa_classifier.ipynb contains the code for all data processing and visualisation in report.pdf, as well as the DeBERTA based classifier.
FLAN_T5_prompting_classifier.ipynb contains the code for the "prompt based" FLAN-T5 classifier.

All training and validation data, including plots for each model in my report can be found in modelDirectory.
I removed the actual model weights due to their file size.