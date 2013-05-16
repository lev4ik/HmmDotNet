using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Predictors
{
    [TestClass]
    public class TrendPredictionErrorEstimatorTest
    {
        private static double[][] _trainingSet;
        private static double[][] _testSet;
        private static double[][] _predictionSet;
        private static int NumberOfComponents = 2;
        private static int NumberOfStates = 4;
        private static int NumberOfIterations = 10;
        private static int LikelihoodTolerance = 20;

        [ClassInitialize]
        public static void Initializer(TestContext context)
        {
            if (_trainingSet == null)
            {
                var util = new TestDataUtils();
                _trainingSet = util.GetSvcData(util.GOOGFilePath, new DateTime(2011, 01, 01), new DateTime(2012, 01, 01));
                _testSet = util.GetSvcData(util.GOOGFilePath, new DateTime(2012, 01, 01), new DateTime(2013, 04, 05));

                var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, NumberOfComponents = NumberOfComponents });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStates) { LogNormalized = true };
                model.Normalized = true;
                model.Train(_trainingSet, NumberOfIterations, LikelihoodTolerance);
                var result = model.Predict(PredictorType.HmmLikelihood, _trainingSet, _testSet, _testSet.Length, NumberOfIterations, LikelihoodTolerance);
                _predictionSet = result.Predicted;
            }
        }

        [TestMethod]
        public void TrendPredictionErrorEstimator_GOOGSeries_TrendPredictionErrorEstimatorCreated()
        {
            var eval = new TrendPredictionErrorEstimator(_testSet, _predictionSet);

            Assert.IsNotNull(eval);
        }

        [TestMethod]
        public void CumulativeForecastError_GOOGSeries_CumulativeForecastErrorCalculated()
        {
            var eval = new TrendPredictionErrorEstimator(_testSet, _predictionSet);
            var cfe = eval.CumulativeForecastError();

            Assert.AreEqual(151.0, cfe[0]);
            Assert.AreEqual(145.0, cfe[1]);
            Assert.AreEqual(147.0, cfe[2]);
            Assert.AreEqual(153.0, cfe[3]);
        }

        [TestMethod]
        public void MeanError_GOOFSeries_MeanErrorCalculated()
        {
            var eval = new TrendPredictionErrorEstimator(_testSet, _predictionSet);
            var me = eval.MeanError();

            Assert.AreEqual(0.48, me[0]);
            Assert.AreEqual(0.46, me[1]);
            Assert.AreEqual(0.47, me[2]);
            Assert.AreEqual(0.49, me[3]);
        }

        [TestMethod]
        public void MeanSquaredError_GOOGSeries_MeanSquaredErrorCalculated()
        {
            var eval = new TrendPredictionErrorEstimator(_testSet, _predictionSet);
            var mse = eval.MeanSquaredError();

            Assert.AreEqual(0.48, mse[0]);
            Assert.AreEqual(0.46, mse[1]);
            Assert.AreEqual(0.47, mse[2]);
            Assert.AreEqual(0.49, mse[3]);
        }

        [TestMethod]
        public void RootMeanSquaredError_GOOGSeries_RootMeanSquaredErrorCalculated()
        {
            var eval = new TrendPredictionErrorEstimator(_testSet, _predictionSet);
            var mse = eval.RootMeanSquaredError();

            Assert.AreEqual(0.69, mse[0]);
            Assert.AreEqual(0.68, mse[1]);
            Assert.AreEqual(0.69, mse[2]);
            Assert.AreEqual(0.7, mse[3]);
        }

        [TestMethod]
        public void MeanAbsoluteDeviation_GOOGSeries_MeanAbsoluteDeviationCalculated()
        {
            var eval = new TrendPredictionErrorEstimator(_testSet, _predictionSet);
            var mad = eval.MeanAbsoluteDeviation();

            Assert.AreEqual(0.48, mad[0]);
            Assert.AreEqual(0.46, mad[1]);
            Assert.AreEqual(0.47, mad[2]);
            Assert.AreEqual(0.49, mad[3]);
        }

        [TestMethod]
        public void MeanAbsolutePercentError_GOOGSeries_MeanAbsolutePercentErrorCalculated()
        {
            var eval = new TrendPredictionErrorEstimator(_testSet, _predictionSet);
            var mape = eval.MeanAbsolutePercentError();

            Assert.AreEqual(47.94, mape[0]);
            Assert.AreEqual(46.03, mape[1]);
            Assert.AreEqual(46.67, mape[2]);
            Assert.AreEqual(48.57, mape[3]);
        }
    }
}
