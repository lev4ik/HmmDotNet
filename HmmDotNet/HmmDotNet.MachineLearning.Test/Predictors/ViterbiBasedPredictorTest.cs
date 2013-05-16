using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.GeneralPredictors.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Predictors
{ 
    [TestClass]
    public class ViterbiBasedPredictorTest
    {
        private const int _NumberOfComponents = 2;
        private const int _NumberOfIterations = 10;
        private const int _NumberOfStates = 2;
        private const int _LikelihoodTolerance = 20;

        [TestMethod]
        public void Predict_HMMMixtureAndFTSESeriesLength1_PredictedOneDay()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var request = new PredictionRequest();
            request.TrainingSet = series;
            request.NumberOfDays = 1;

            var pred = new ViterbiBasedPredictor();
            var res = pred.Predict(model, request);

            Assert.AreEqual(1, res.Predicted.Length);
        }

        [TestMethod]
        public void Predict_HMMMixtureAndFTSESeriesLength20_Predicted20Days()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);

            var pred = new ViterbiBasedPredictor();
            var request = new PredictionRequest { TrainingSet = series, NumberOfDays = 20, TestSet = test };
            pred.NumberOfIterations = _NumberOfIterations;
            pred.LikelihoodTolerance = _LikelihoodTolerance;
            var predictions = pred.Predict(model, request);

            Assert.AreEqual(20, predictions.Predicted.Length);            
        }

        [TestMethod]
        public void Predict_HMMMultivariateAndFTSESeriesLength1_PredictedOneDay()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var request = new PredictionRequest();
            request.TrainingSet = series;
            request.NumberOfDays = 1;

            var pred = new ViterbiBasedPredictor();
            var res = pred.Predict(model, request);

            Assert.AreEqual(1, res.Predicted.Length);
        }

        [TestMethod]
        public void Predict_HMMMultivariateAndFTSESeriesLength20_Predicted20Days()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);

            var pred = new ViterbiBasedPredictor();
            var request = new PredictionRequest { TrainingSet = series, NumberOfDays = 20, TestSet = test };
            pred.NumberOfIterations = _NumberOfIterations;
            pred.LikelihoodTolerance = _LikelihoodTolerance;
            var predictions = pred.Predict(model, request);

            Assert.AreEqual(20, predictions.Predicted.Length);      
        }

        [TestMethod]
        public void Evaluate_HMMMixtureAndFTSESeriesLength20_ErrorEstimatorCalculated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);

            var pred = new ViterbiBasedPredictor();
            var request = new PredictionRequest { TrainingSet = series, NumberOfDays = 20, TestSet = test };
            pred.NumberOfIterations = _NumberOfIterations;
            pred.LikelihoodTolerance = _LikelihoodTolerance;
            var predictions = pred.Predict(model, request);
            var errorRequest = new EvaluationRequest
                {
                    EstimatorType = ErrorEstimatorType.Value,
                    PredictionParameters = request,
                    PredictionToEvaluate = predictions
                };

            var errorEstimation = pred.Evaluate(errorRequest);
            for (int i = 0; i < series[0].Length; i++)
            {
                Assert.IsTrue(errorEstimation.CumulativeForecastError[i] > 0);
                Assert.IsTrue(errorEstimation.MeanAbsoluteDeviation[i] > 0);
                Assert.IsTrue(errorEstimation.MeanAbsolutePercentError[i] > 0);
                Assert.IsTrue(errorEstimation.MeanError[i] > 0);
                Assert.IsTrue(errorEstimation.MeanSquaredError[i] > 0);
                Assert.IsTrue(errorEstimation.RootMeanSquaredError[i] > 0);
            }
        }

        [TestMethod]
        public void Evaluate_HMMMultivariateAndFTSESeriesLength20_ErrorEstimatorCalculated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);

            var pred = new ViterbiBasedPredictor();
            var request = new PredictionRequest { TrainingSet = series, NumberOfDays = 20, TestSet = test };
            pred.NumberOfIterations = _NumberOfIterations;
            pred.LikelihoodTolerance = _LikelihoodTolerance;
            var predictions = pred.Predict(model, request);
            var errorRequest = new EvaluationRequest
            {
                EstimatorType = ErrorEstimatorType.Value,
                PredictionParameters = request,
                PredictionToEvaluate = predictions
            };

            var errorEstimation = pred.Evaluate(errorRequest);
            for (int i = 0; i < series[0].Length; i++)
            {
                Assert.IsTrue(errorEstimation.CumulativeForecastError[i] > 0);
                Assert.IsTrue(errorEstimation.MeanAbsoluteDeviation[i] > 0);
                Assert.IsTrue(errorEstimation.MeanAbsolutePercentError[i] > 0);
                Assert.IsTrue(errorEstimation.MeanError[i] > 0);
                Assert.IsTrue(errorEstimation.MeanSquaredError[i] > 0);
                Assert.IsTrue(errorEstimation.RootMeanSquaredError[i] > 0);
            }
        }
    }
}
