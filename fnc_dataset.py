#built upon code from  https://github.com/FakeNewsChallenge/fnc-1-baseline
import pandas as pd
from sklearn.model_selection import train_test_split


#custom dataset class for easy access to the dataset, distinct from the huggingface dataset class
class FNCDataset():
    def __init__(self, stances, articles):
        self.stances = stances
        self.articles = articles

        #split stances into related and unrelated
        self.relatedStances = self.stances[self.stances["Stance"] != "unrelated"]
        self.unrelatedStances = self.stances[self.stances["Stance"] == "unrelated"]

        #split the articles into related and unrelated, construct a unique set and set which allows duplicate articles
        relatedBodyIds = self.relatedStances["Body ID"]
        unrelatedBodyIds = self.unrelatedStances["Body ID"]
        uniqueRelatedBodyIds = relatedBodyIds.unique()
        uniqueUnrelatedBodyIds = unrelatedBodyIds.unique()

        self.relatedBodies = self.articles.loc[relatedBodyIds]
        self.unrelatedBodies = self.articles.loc[unrelatedBodyIds]
        self.uniqueRelatedBodies = self.articles.loc[uniqueRelatedBodyIds]
        self.uniqueUnrelatedBodies = self.articles.loc[uniqueUnrelatedBodyIds]

        #combine the stances and articles into a single dataframe
        self.headlinesBodiesCombined = self.stances.merge(self.articles, left_on="Body ID", right_index=True)

    @classmethod
    def from_csv(cls, labelledStancesPath, articlesPath):
        stances = pd.read_csv(labelledStancesPath)
        stances['Body ID'] = stances['Body ID'].astype(int)

        articles = pd.read_csv(articlesPath)
        articles['Body ID'] = articles['Body ID'].astype(int)
        articles.set_index("Body ID", inplace=True)
        
        return cls(stances, articles)

    def split(self, testSize, randomState=None):
        #split the stances into training and testing sets, if randomState
        #stratify ensures the distribution of stances for the training and testing sets are the same as the original dataset
        trainStances, testStances = train_test_split(self.stances, test_size=testSize, stratify=self.stances["Stance"], random_state=randomState)

        #select corresponding articles for testing and training sets
        trainBodyIds = trainStances["Body ID"].unique()
        testBodyIds = testStances["Body ID"].unique()
        trainArticles = self.articles.loc[trainBodyIds]
        testArticles = self.articles.loc[testBodyIds]

        #create and return testing and training datasets
        trainInstance = FNCDataset(trainStances, trainArticles)
        testInstance = FNCDataset(testStances, testArticles)
        return trainInstance, testInstance

