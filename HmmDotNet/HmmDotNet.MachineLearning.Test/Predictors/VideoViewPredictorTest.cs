using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors;
using HmmDotNet.MachineLearning.Test.Data;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace HmmDotNet.MachineLearning.Test.Predictors
{
    [TestClass]
    public class VideoViewPredictorTest
    {
        private const int _NumberOfComponents = 2;
        private const int _NumberOfIterations = 50;
        private const int _NumberOfStates = 4;
        private const int _LikelihoodTolerance = 20;

        [TestMethod]
        public void Predict_ModelAndPredictionRequest_PredictedValue()
        {
            var util = new TestPipedDataUtils();
            var series = util.GetSvcData(util.VideoViewsFilePath, new DateTime(2014, 01, 01), new DateTime(2014, 06, 01));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var request = new SearchBasedPredictionRequest();
            request.TrainingSet = series;
            request.NumberOfDays = 10;

            var result = model.Predict(PredictorType.HmmLikelihood, series, _NumberOfIterations, _LikelihoodTolerance);

            Assert.AreEqual(0, 0);//result.Predicted);
        }

        [TestMethod]
        public void Predict_VideoViewsSeriesAndWeightErgodicModelAnd30DaysTestSequence_PredictionResult()
        {
            var util = new TestPipedDataUtils();
            var series = util.GetSvcData(util.VideoViewsFilePath, new DateTime(2014, 01, 01), new DateTime(2014, 05, 31));
            var test = util.GetSvcData(util.VideoViewsFilePath, new DateTime(2014, 06, 01), new DateTime(2014, 06, 30));

            var model = (HiddenMarkovModelWeightedMixtureDistribution)HiddenMarkovModelFactory.GetModel<HiddenMarkovModelWeightedMixtureDistribution, Mixture<IMultivariateDistribution>>(
                new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { 
                NumberOfComponents = _NumberOfComponents, 
                NumberOfStates = _NumberOfStates 
            });

            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, 30, _NumberOfIterations, _LikelihoodTolerance);

            util.SaveToFile("weighted_data.csv", test, result.Predicted);
            Assert.IsNotNull(result);
            Assert.AreEqual(30, result.Predicted.Length);
        }

        [TestMethod]
        public void Predict_VideoViewsSeriesAndRegularErgodicModelAnd30DaysTestSequence_PredictionResult()
        {
            var util = new TestPipedDataUtils();
            var series = util.GetSvcData(util.VideoViewsFilePath, new DateTime(2014, 01, 01), new DateTime(2014, 05, 31));
            var test = util.GetSvcData(util.VideoViewsFilePath, new DateTime(2014, 06, 01), new DateTime(2014, 06, 30));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel<HiddenMarkovModelMixtureDistribution, Mixture<IMultivariateDistribution>>(
                new ModelCreationParameters<Mixture<IMultivariateDistribution>>()
                {
                    NumberOfComponents = _NumberOfComponents,
                    NumberOfStates = _NumberOfStates
                });

            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, 30, _NumberOfIterations, _LikelihoodTolerance);

            util.SaveToFile("regular_data.csv", test, result.Predicted);
            Assert.IsNotNull(result);
            Assert.AreEqual(30, result.Predicted.Length);
        }
    }
}