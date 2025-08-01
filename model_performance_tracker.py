import pandas as pd


class ModelPerformanceTracker:
    def __init__(self, numLabels):
        self.numLabels = numLabels
        self.reset_epoch()
        self.metricsList = []

    def reset_epoch(self):
        #class specific loss and accuracy
        self.classCorrect = {label: 0 for label in range(self.numLabels)}
        self.classTotal = {label: 0 for label in range(self.numLabels)}
        self.classLoss = {label: 0.0 for label in range(self.numLabels)}

        #used for calculating precision, recall and f1 score
        self.truePositives = {label: 0 for label in range(self.numLabels)}
        self.falsePositives = {label: 0 for label in range(self.numLabels)}
        self.falseNegatives = {label: 0 for label in range(self.numLabels)}

        #overall loss
        self.totalLoss = 0.0

    def update_batch(self, labels, predicted, loss, batchSize):
        #update loss and accuracy from a single new batch, done individually for each class
        for label in range(self.numLabels):
            mask = (labels == label)
            correctPredictions = (predicted[mask] == labels[mask]).sum().item()

            #update class specific loss and accuracy
            self.classCorrect[label] += correctPredictions
            self.classTotal[label] += mask.sum().item()
            self.classLoss[label] += loss.item() * mask.sum().item() / batchSize

            #update true positives, false positives and false negatives
            self.truePositives[label] += correctPredictions
            self.falsePositives[label] += ((predicted == label) & (labels != label)).sum().item()
            self.falseNegatives[label] += ((predicted != label) & (labels == label)).sum().item()

        #update overall loss
        self.totalLoss += loss.item()

    def finish_epoch(self, epoch):
        #calculate class specific accuracy and loss, also f1 score with equal class weighting
        classAccuracy = {}
        classAvgLoss = {}
        classPrecision = {}
        classRecall = {}
        classF1 = {}

        for label in range(self.numLabels):
            #class specific accuracy and loss
            classAccuracy[label] = self.classCorrect[label] / self.classTotal[label]
            classAvgLoss[label] = self.classLoss[label] / self.classTotal[label]

            #precision, recall, and F1 score
            classPrecision[label] = self.truePositives[label] / (self.truePositives[label] + self.falsePositives[label])
            classRecall[label] = self.truePositives[label] / (self.truePositives[label] + self.falseNegatives[label])
            classF1[label] = 2 * classPrecision[label] * classRecall[label] / (classPrecision[label] + classRecall[label])

        #compute f1 score with an equal weighting per class
        f1Score = sum(classF1.values()) / self.numLabels

        #calculate overall accuracy and loss
        totalCorrect = sum(self.classCorrect.values())
        totalSamples = sum(self.classTotal.values())
        overallAccuracy = totalCorrect / totalSamples
        overallAvgLoss = self.totalLoss / totalSamples

        #package epoch metrics into a dictionary and add to list
        epochMetrics = {
            "epoch": epoch,
            "overallAccuracy": overallAccuracy,
            "overallLoss": overallAvgLoss,
            "f1Score": f1Score
        }
        for label in range(self.numLabels):
            epochMetrics[f"classAccuracy_{label}"] = classAccuracy[label]
            epochMetrics[f"classLoss_{label}"] = classAvgLoss[label]

        self.metricsList.append(epochMetrics)

    def get_metrics_dataframe(self):
        #convert the metrics list into a dataframe and return
        metricsDataframe = pd.DataFrame(self.metricsList)
        metricsDataframe["epoch"] = metricsDataframe["epoch"].astype(int)
        return metricsDataframe

