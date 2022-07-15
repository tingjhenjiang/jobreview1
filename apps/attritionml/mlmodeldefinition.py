import pandas as pd
import numpy as np
import sklearn,pickle
import sys, pathlib
import sklearn.ensemble, sklearn.metrics, sklearn.feature_selection, sklearn.preprocessing, sklearn.decomposition
import sklearn.model_selection, sklearn.utils, sklearn.linear_model, sklearn.pipeline, sklearn.manifold
import sklearn.naive_bayes,sklearn.discriminant_analysis,sklearn.base
import tensorflow as tf
#import condor_tensorflow as condor
#math, scipy

class Sampling(tf.keras.layers.Layer):
    def __init__(self,
                 name='vaeSampling',
                 **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        z_mean, z_log_var = inputs
        nshape = tf.keras.backend.shape(z_mean)
        epsilon = tf.keras.backend.random_normal(shape=nshape)
        return z_mean + tf.keras.backend.exp(z_log_var/2) * epsilon #tf.exp(0.5 * z_log_var)

class vae_loss(tf.keras.losses.Loss):
    def __init__(self,
                 *args,
                 #encoder_mu, #y_true
                 #encoder_log_variance, #y_predict
                 name='vae_loss',
                 **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_predict):
        return self.vae_loss(y_true, y_predict)

    def vae_reconstruction_loss(self, y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(self, encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(self, y_true, y_predict):
        encoder_mu = self.encoder_mu
        encoder_log_variance = self.encoder_log_variance
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(self, y_true, y_predict):
        reconstruction_loss = self.vae_reconstruction_loss(y_true, y_predict)
        kl_loss = self.vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + kl_loss
        return loss

def getVaeEncoder(latent_dim=16, srcdfDims=68):
    encoder_inputs = tf.keras.Input(shape=(srcdfDims,))
    denselayersEncoder = tf.keras.Sequential([
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(16, kernel_initializer='lecun_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
    ], name='denselayersEncoder')
    x = denselayersEncoder(encoder_inputs)
    encoder_mu = tf.keras.layers.Dense(latent_dim, name="z_mean", kernel_initializer='lecun_normal')(x)
    encoder_log_variance = tf.keras.layers.Dense(latent_dim, name="z_log_var", kernel_initializer='lecun_normal')(x)
    encoder_output = Sampling(name="encoder_output")([encoder_mu, encoder_log_variance])# Sampling()([encoder_mu, encoder_log_variance])
    encoder = tf.keras.Model(encoder_inputs, encoder_output, name="encoderModel")
    return {'model':encoder,'encoder_mu':encoder_mu,'encoder_log_variance':encoder_log_variance}

def getVaeDecoder(latent_dim=16, srcdfDims=68):
    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    denselayersDecoder = tf.keras.Sequential([
        tf.keras.layers.Dense(16, kernel_initializer='lecun_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
    ], name='denselayersDecoder')
    x = denselayersDecoder(decoder_inputs)
    decoder_outputs = tf.keras.layers.Dense(srcdfDims, kernel_initializer='lecun_normal')(x)
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoderModel")
    return decoder

def getVaeModel(latent_dim=16, srcdfDims=68):
    vae_input = tf.keras.layers.Input(shape=(srcdfDims,), name="VAE_input")
    encoders = getVaeEncoder(latent_dim=latent_dim, srcdfDims=srcdfDims)
    vae_encoder_output = encoders['model'](vae_input)
    vae_decoder_output = getVaeDecoder(latent_dim=latent_dim, srcdfDims=srcdfDims)(vae_encoder_output)
    vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAEModel")
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=vae_loss(encoders['encoder_mu'], encoders['encoder_log_variance']))
    #vae.summary()
    return vae

class AttritionClassifier():
    def __init__(self):
        testInputDfFile = (pathlib.Path() / 'attritionproject.csv').resolve()
        srcdfX = pd.read_csv(testInputDfFile).drop(columns=['Attrition'])
        self.contiVars = ['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        self.ordinalVars = ['Education','JobInvolvement','JobLevel','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance']
        self.categoricalVars = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        self.preprocessorsParameters = self.reorderDFcolsAndlearnPreprocessors(srcdfX)
        clfModelFile = (pathlib.Path() / 'apps' / 'attritionml' / 'attritionClassifier.pkl').resolve()
        with open(clfModelFile, 'rb') as f:
            self.clfmodel = pickle.load(f)
            f.close()

    def reorderDFcolsAndlearnPreprocessors(self, inputDF, ret_preprocessor=True):
        newDF = inputDF.loc[:,self.contiVars+self.ordinalVars]
        scaler = sklearn.preprocessing.RobustScaler().fit(newDF)
        if False:
            ovPreprocessors = {}
            newOvcolnames = []
            for ov in self.ordinalVars:
                newOvcolnames.extend(['{}{}'.format(ov,i) for i in range(1,len(inputDF[ov].unique()))])
                ovPreprocessors[ov] = condor.CondorOrdinalEncoder(nclasses=len(inputDF[ov].unique())).fit(inputDF[ov]-inputDF[ov].min())
        onehotPreprocessor = sklearn.preprocessing.OneHotEncoder(sparse=False).fit(inputDF.loc[:,self.categoricalVars])
        for vars in [self.categoricalVars]:
            for colname in vars:
                newDF[colname] = inputDF[colname]
        returnDict = {}
        returnDict['newTraingDF'] = newDF.reset_index(drop=True)
        if ret_preprocessor:
            returnDict['scaler'] = scaler
            #returnDict['ovPreprocessors'] = ovPreprocessors
            returnDict['onehotPreprocessor'] = onehotPreprocessor
            #returnDict['newOvcolnames'] = newOvcolnames
        return returnDict

    def attritionPreprocess(self, inputData):
        if isinstance(inputData, dict):
            srcdf = pd.DataFrame(inputData, index=[0])
        else:
            srcdf = inputData
        srcdf = srcdf.drop(columns=['Over18','EmployeeCount','StandardHours'])

        #ordinal variable encoding
        if False:
            ovDFs = []
            for ov in self.ordinalVars:
                ovV = self.preprocessorsParameters['ovPreprocessors'][ov].transform(srcdf[ov])
                ovDFs.append(pd.DataFrame(ovV))
            ovDFs = pd.concat(ovDFs, axis=1, ignore_index=True).reset_index(drop=True)
            ovDFs.columns = self.preprocessorsParameters['newOvcolnames']

        #scale standardization
        realcontiVarsDF = pd.DataFrame(
                                data=self.preprocessorsParameters['scaler'].transform(srcdf.loc[:,self.contiVars+self.ordinalVars]),
                                columns=self.contiVars+self.ordinalVars
                                )
        
        #one-hot encoding
        #sys.exit(type(self.preprocessorsParameters['onehotPreprocessor'].transform(srcdf.loc[:,self.categoricalVars])))
        catgVarsDF = pd.DataFrame(
                        data = self.preprocessorsParameters['onehotPreprocessor'].transform(srcdf.loc[:,self.categoricalVars]),
                        columns=self.preprocessorsParameters['onehotPreprocessor'].get_feature_names_out()
                        )

        cleanedsrcdf = realcontiVarsDF#pd.concat([realcontiVarsDF, ovDFs, catgVarsDF_dummied], axis=1).reset_index(drop=True)
        for concatdf in [catgVarsDF]:
            for col in concatdf.columns:
                cleanedsrcdf[col] = concatdf[col]

        for col in cleanedsrcdf.columns:
            if isinstance(cleanedsrcdf[col],pd.core.frame.DataFrame):
                tempseries = cleanedsrcdf[col].reset_index(drop=True).iloc[:,0]
                cleanedsrcdf = cleanedsrcdf.drop(columns=[col])
                cleanedsrcdf[col] = tempseries

        cleanedsrcdf = pd.get_dummies(cleanedsrcdf).astype('float64')

        return cleanedsrcdf

    def getVarsAccordingToType(self, sdf, dtype='continueous', ret='col'):
        if dtype in ['continueous','conti']:
            needCols = [col for col in sdf.columns if sdf[col].dtype not in ['object','category']]
        else:
            needCols = [col for col in sdf.columns if sdf[col].dtype in ['object','category']]
        if ret in ['val','value','values']:
            return sdf.loc[:,needCols]
        else:
            return needCols

    def predict(self, input_data):
        return self.clfmodel.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = "No"
        if input_data[1] > 0.5:
            label = "Yes"
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.attritionPreprocess(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction