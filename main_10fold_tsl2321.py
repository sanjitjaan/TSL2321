import pandas as pd
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation, PredictionOutput
import weka.core.converters as converters
from weka.core.classes import Random
jvm.start()
import math
from zipfile import ZipFile
import glob

try:
    trainData = converters.load_any_file('../data/input_encoded_sorted.csv')
    trainData.class_is_last()
    #print(data)
    hybridList = ["weka.classifiers.meta.Bagging", "weka.classifiers.meta.RandomCommittee"]
    standaloneList = ["weka.classifiers.functions.MultilayerPerceptron", "weka.classifiers.trees.REPTree"]
    for hybrid in hybridList:
        for standalone in standaloneList:
            try:
                cls = Classifier(classname=hybrid, options=["-W", standalone])
                pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")

                evl = Evaluation(trainData)
                evl.crossvalidate_model(cls, trainData, 10, Random(1), pout)
                print('Model Name: '+hybrid+'-'+standalone)
                print(evl.summary())
                #print(pout.buffer_content())

                df = pd.DataFrame(columns=['inst#','actual','predicted','error'])
                for i,row in enumerate(pout.buffer_content().split('\n')[:-1]):
                    if i==0:
                        continue
                    df.loc[i]=row.split(',')
                df.to_csv('../results/'+hybrid.split('.')[-1]+'-'+standalone.split('.')[-1]+'_%02df_result.csv'%10, index=False)
            except:
                continue
    for standalone in standaloneList:
        try:
            cls = Classifier(classname=standalone)
            pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")

            evl = Evaluation(trainData)
            evl.crossvalidate_model(cls, trainData, 10, Random(1), pout)
            print('Model Name: '+standalone)
            print(evl.summary())

            df = pd.DataFrame(columns=['inst#','actual','predicted','error'])
            for i,row in enumerate(pout.buffer_content().split('\n')[:-1]):
                if i==0:
                    continue
                df.loc[i]=row.split(',')
            df.to_csv('../results/'+standalone.split('.')[-1]+'_%02df_result.csv'%10, index=False)
        except:
            continue
except:
    jvm.stop()
jvm.stop()