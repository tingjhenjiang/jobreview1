# %%
import sys,copy,re,importlib,pathlib,json
import pandas as pd
import numpy as np
import tensorflow as tf
import math, scipy, sys
from IPython.display import display
from functools import reduce
import pathlib
import sklearn
import matplotlib.pyplot as plt
#import plotly
#import plotly.express as px
import sklearn.ensemble, sklearn.metrics, sklearn.feature_selection, sklearn.preprocessing, sklearn.decomposition
import sklearn.model_selection, sklearn.utils, sklearn.linear_model, sklearn.pipeline, sklearn.manifold
import sklearn.naive_bayes,sklearn.discriminant_analysis,sklearn.base,sklearn.compose,sklearn.neural_network
import dython,pickle
import argparse
#import mlmodeldefinition
#importlib.reload(mlmodeldefinition)
pd.set_option('display.max_columns', 5000)
try:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(str(e))

# %%
def generateSrcData():
    testInputDfFile = (pathlib.Path() / 'attritionproject.csv').resolve()
    srcdf = pd.read_csv(testInputDfFile)

    # %% [markdown]
    # ## 檢查遺漏值，區別變數型態

    # %%
    continueousVars = []
    categoricalVars = []
    for col in srcdf.columns:
        if col=='Attrition':
            pass
        elif srcdf[col].dtype in ['object','category']:
            categoricalVars.append(col)
        else:
            continueousVars.append(col)


    # %%
    srcdf = pd.read_csv(testInputDfFile)
    srcdfY = srcdf.loc[:,'Attrition']
    lbzedSrcdfY = sklearn.preprocessing.LabelBinarizer().fit(srcdfY)
    lbzedSrcdfY = lbzedSrcdfY.transform(srcdfY)[:,0]

    srcdf = srcdf.drop(columns=['Attrition'])


    checkvars = ['Over18','EmployeeCount','StandardHours']
    cleanedsrcdf = {'0dropped':srcdf.drop(columns=checkvars)}
    try:
        continueousVars.remove('EmployeeCount')
        continueousVars.remove('StandardHours')
        categoricalVars.remove('Over18')
        categoricalVars.remove('BusinessTravel')
    except:
        pass

    #次序資料
    #重新編碼BusinessTravel欄位
    cleanedsrcdf['0dropped'] = cleanedsrcdf['0dropped'].replace({'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2})
    ordinalVarsLevels = {'BusinessTravel':3,'Education':5,'EnvironmentSatisfaction':4,'JobInvolvement':4,'JobLevel':5,'JobSatisfaction':4,'PerformanceRating':4,'RelationshipSatisfaction':4,'StockOptionLevel':4,'WorkLifeBalance':4}
    ordinalVars = list(ordinalVarsLevels.keys())
    realContiVars = [col for col in continueousVars if col not in ordinalVars]

    cleanedsrcdf['0dropped1ordinaled'] = cleanedsrcdf['0dropped'].loc[:,:]
    cleanedsrcdf['0dropped1ordinaled'].loc[:,ordinalVars] = cleanedsrcdf['0dropped1ordinaled'].loc[:,ordinalVars]-cleanedsrcdf['0dropped1ordinaled'].loc[:,ordinalVars].min()
    ordinalColumnTransformers = []
    for col,levels in ordinalVarsLevels.items():
        #累積機率的encoding
        cumulativeOrdinalTransformer = (col+'Condor',mlmodeldefinition.CustomCondorOrdinalEncoder(nclasses=levels),col)
        ordinalColumnTransformers.append(cumulativeOrdinalTransformer)
    coltransformer = sklearn.compose.ColumnTransformer(ordinalColumnTransformers, remainder='passthrough', verbose_feature_names_out=False).fit(cleanedsrcdf['0dropped1ordinaled'])
    cleanedsrcdf['0dropped1ordinaled'] = pd.DataFrame(data=coltransformer.transform(cleanedsrcdf['0dropped1ordinaled']),columns=coltransformer.get_feature_names_out())
    convertedOrdinalVars = [col for col in coltransformer.get_feature_names_out() if col not in realContiVars+categoricalVars]


    # %%
    scalerModel = {}
    for previousTestd in copy.deepcopy(list(cleanedsrcdf.keys())): #to avoid dict change error
        cleanedsrcdf[previousTestd+'2rbscaled'] = cleanedsrcdf[previousTestd].loc[:,:]
        scalerModel[previousTestd] = sklearn.preprocessing.RobustScaler().fit(cleanedsrcdf[previousTestd+'2rbscaled'].loc[:, realContiVars])
        cleanedsrcdf[previousTestd+'2rbscaled'][realContiVars] = pd.DataFrame(
            data=scalerModel[previousTestd].transform(cleanedsrcdf[previousTestd+'2rbscaled'].loc[:, realContiVars]),
            columns=realContiVars
            ).astype('float64')
        cleanedsrcdf[previousTestd][realContiVars] = cleanedsrcdf[previousTestd].loc[:, realContiVars].astype('float64')    

        if re.search('ordinaled',previousTestd)!=None:
            cleanedsrcdf[previousTestd+'2rbscaled'][convertedOrdinalVars] = cleanedsrcdf[previousTestd+'2rbscaled'].loc[:, convertedOrdinalVars].astype('float64')
            cleanedsrcdf[previousTestd][convertedOrdinalVars] = cleanedsrcdf[previousTestd].loc[:, convertedOrdinalVars].astype('float64')
        else:
            cleanedsrcdf[previousTestd][ordinalVars] = cleanedsrcdf[previousTestd].loc[:, ordinalVars].astype('float64')

    # %%
    attrition_classWeights = sklearn.utils.class_weight.compute_class_weight('balanced',classes=np.unique(lbzedSrcdfY),y=lbzedSrcdfY)
    attrition_classWeights_dict = dict(zip(np.unique(lbzedSrcdfY),attrition_classWeights))
    attrition_sampleweights = sklearn.utils.class_weight.compute_sample_weight(attrition_classWeights_dict, y=lbzedSrcdfY)


# %%
def ind_tfkerasClf(X, y=None, kwargs=None):

    def convertKerasFuncs(paramdict):
        tempres = None
        if 'callback' in paramdict:
            if paramdict['callback']=='EarlyStopping':
                return tf.keras.callbacks.EarlyStopping(**paramdict['parameters'])
        if 'metric' in paramdict:
            if paramdict['metric']=='AUC':
                return tf.keras.metrics.AUC(**paramdict['parameters'])
            elif paramdict['metric']=='Recall':
                return tf.keras.metrics.Recall(**paramdict['parameters'])
        if 'loss' in paramdict:
            if paramdict['loss']=='binary_crossentropy':
                return tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy')
            elif paramdict['loss']=='binary_focal_crossentropy':
                return tf.keras.losses.BinaryFocalCrossentropy(name='binary_focal_crossentropy')
        if 'optimizer' in paramdict:
            if paramdict['optimizer']=='Adam':
                return tf.keras.optimizers.Adam(**paramdict['parameters'])
        if 'activation' in paramdict:
            if paramdict['activation']=='relu':
                return tf.keras.activations.relu
            if paramdict['activation']=='sigmoid':
                return tf.keras.activations.sigmoid
        return paramdict
    def filterOutEmptyElement(srclist):
        return [l for l in srclist if l not in [None]]
    def getConvertedKerasFuncsList(srclist):
        if isinstance(srclist, list): #'A list found'
            needfuncs = [convertKerasFuncs(d) for d in srclist]
            needfuncs = filterOutEmptyElement(needfuncs)
        else: #'A non-list found'
            needfuncs = convertKerasFuncs(srclist)
        return needfuncs
    def convertKwarg(kwargs):
        import copy
        convertedkwargs = copy.deepcopy(kwargs)
        for key in ['callbacks','metrics','weighted_metrics','loss','optimizer','activation','lastLayerActivation']:
            if key in convertedkwargs:
                try:
                    convertedkwargs[key] = getConvertedKerasFuncsList(kwargs[key])
                except Exception as e:
                    convertedkwargs[key] = kwargs[key]
                    print(f'error in {key} for {e}')
        print('kwargs is {}'.format(kwargs))
        print('convertedkwargs is {}'.format(convertedkwargs))
        return convertedkwargs
    def newXandSampleWeight(X):
        if sample_weight==True:
            sample_weight = X['sampleweight']
        else:
            sample_weight = None
        X = X.drop(columns='sampleweight')
        sample_weight_values = sample_weight
        return X, sample_weight
    def buildModel(X,modelKwargs=None):
        forseqlayers = [
            [tf.keras.layers.Dense(
                units=layerunits,
                activation=modelKwargs['activation'],
                activity_regularizer=tf.keras.regularizers.L2(modelKwargs['l2Alpha']),
                name='dense{}'.format(li)
            ),
            tf.keras.layers.BatchNormalization(name='batchnormalization{}'.format(li))
            ]
            for li,layerunits in enumerate(modelKwargs['denseLayerUnits'])
        ]
        forseqlayers = reduce(lambda x,y:x+y,forseqlayers)
        forseqlayers = [tf.keras.layers.Input(shape=X.shape[1:])]+forseqlayers
        forseqlayers.append(tf.keras.layers.Dropout(rate=modelKwargs['dropout_rate']))
        forseqlayers.append(tf.keras.layers.Dense(units=modelKwargs['lastLayerUnits'],activation=modelKwargs['lastLayerActivation']))
        forseqlayers = tf.keras.Sequential(forseqlayers)
        model = forseqlayers#tf.keras.models.Model(inputs=inputlayer, outputs=output)
        model.compile(optimizer=modelKwargs['optimizer'], loss=modelKwargs['loss'], metrics=modelKwargs['metrics'], weighted_metrics=modelKwargs['weighted_metrics'])
        return model
        #self.model.summary()
        #sys.exit()
    def fit(X, y=None, kwargs=None):
        modelKwargs = convertKwarg(kwargs)
        fittingOnlyKwargs = {k:v for k,v in modelKwargs.items() if k in [
            'batch_size','epochs','verbose','callbacks','validation_split','validation_data','shuffle','class_weight','sample_weight','initial_epoch',
            'steps_per_epoch','validation_steps','validation_batch_size','validation_freq','max_queue_size','workers','use_multiprocessing'
        ]}
        #    'batch_size': self.modelKwargs['batch_size'],
        #    'epochs': self.modelKwargs['epochs'],
        #    'validation_split': self.modelKwargs['validation_split'],
        #    'callbacks': self.modelKwargs['callbacks'],
        #}
        #allkwargs = {**kwargs,**additionalKwargs}
        model = buildModel(X,modelKwargs)
        tf.keras.backend.clear_session()
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except Exception as e:
            print(str(e))
        model.fit(X, y, **fittingOnlyKwargs)
        return model
    def predict(X, y=None, model=None):
        X = X.drop(columns='sampleweight')
        y_pred = model.predict(X)
        y_pred = y_pred.argmax(axis=-1)
        return y_pred
    def predict_proba(X, model=None):
        return model.predict(X)

    from datetime import datetime
    import time
    ts = datetime.timestamp(datetime.now())
    filepath = (pathlib.Path()/f'{ts}.h5').resolve()
    model = fit(X, y, kwargs)
    tf.keras.models.save_model(model, filepath=filepath)
    return (ts,filepath)

def loadmodelAndPredict(X, modelfilepath):
    model = tf.keras.models.load_model(modelfilepath)
    y_pred = model.predict(X)
    return y_pred

#parser = argparse.ArgumentParser("simple_example")
#parser.add_argument("-i", "--input", help="A dict for model building and training.", type=json.loads)
#args = parser.parse_args()

#if __name__ == "__main__":
#    print(args.input)
