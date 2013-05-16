using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.HiddenMarkovModel
{
    [TestClass]
    public class HiddenMarkovModelMixtureDistributionTest
    {
        private const int _NumberOfComponents = 2;
        private const int _NumberOfStates = 2;
        private const int _NumberOfStatesRightLeft = 4;
        private const int _NumberOfIterations = 10;
        private const int _LikelihoodTolerance = 20;
        private const int _Delta = 3;

        private NormalDistribution[] CreateEmissions(double[][] observations, int numberOfEmissions)
        {
            var emissions = new NormalDistribution[numberOfEmissions];
            // Create initial emmissions , TMP and Pi are already created
            var algo = new KMeans();
            algo.CreateClusters(observations, numberOfEmissions, KMeans.KMeansDefaultIterations, (numberOfEmissions > 3) ? InitialClusterSelectionMethod.Furthest : InitialClusterSelectionMethod.Random);

            for (int i = 0; i < numberOfEmissions; i++)
            {
                var mean = algo.ClusterCenters[i];
                var covariance = algo.ClusterCovariances[i];

                emissions[i] = new NormalDistribution(mean, covariance);
            }

            return emissions;
        }

        [TestMethod]
        public void HiddenMarkovModelMixtureDistribution_NumberOfStates_ErgodicModelCreated()
        {
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents ,NumberOfStates);

            Assert.AreEqual(ModelType.Ergodic, model.Type);
            Assert.AreEqual(_NumberOfStates, model.N);            
            Assert.IsFalse(model.Normalized);
            Assert.IsNull(model.Emission);
            for (int n = 0; n < _NumberOfStates; n++)
            {
                Assert.AreEqual(0.5, model.Pi[n]);
                for (int i = 0; i < _NumberOfStates; i++)
                {
                    Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[n][i]);
                }
            }
        }

        [TestMethod]
        public void HiddenMarkovModelMixtureDistribution_NumberOfStatesAndDelta_RightLeftModelCreated()
        {
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStatesRightLeft, Delta = _Delta });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStatesRightLeft, Delta);

            Assert.AreEqual(ModelType.LeftRight, model.Type);
            Assert.AreEqual(_NumberOfStatesRightLeft, model.N);
            Assert.IsFalse(model.Normalized);
            Assert.IsNull(model.Emission);
            Assert.AreEqual(1d, model.Pi[0]);
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][0]);
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][1]);
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][2]);
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][3]);
            Assert.AreEqual(0d, model.TransitionProbabilityMatrix[1][0]);
            Assert.AreEqual(1d / 3d, model.TransitionProbabilityMatrix[1][1]);
            Assert.AreEqual(1d / 3d, model.TransitionProbabilityMatrix[1][2]);
            Assert.AreEqual(1d / 3d, model.TransitionProbabilityMatrix[1][3]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[2][0]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[2][1]);
            Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[2][2]);
            Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[2][3]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[3][0]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[3][1]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[3][2]);
            Assert.AreEqual(1d, model.TransitionProbabilityMatrix[3][3]);
        }

        [TestMethod]
        public void HiddenMarkovModelMixtureDistribution_ModelState_ModelCreated()
        {
            var pi = new double[_NumberOfStates] { 0.5, 0.5 };
            var tpm = new double[_NumberOfStates][] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } };
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var distributions = CreateEmissions(series, _NumberOfStates * _NumberOfComponents);
            var emissions = new Mixture<IMultivariateDistribution>[_NumberOfStates];
            for (int i = 0; i < _NumberOfStates; i++)
            {
                emissions[i] = new Mixture<IMultivariateDistribution>(_NumberOfComponents, series[0].Length);
                for (int j = 0; j < _NumberOfComponents; j++)
                {
                    emissions[i].Components[j] = distributions[j + _NumberOfComponents * i];
                }
            }

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { Pi = pi, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModelMixtureDistribution(pi, tpm, emissions);

            Assert.AreEqual(ModelType.Ergodic, model.Type);
            Assert.AreEqual(_NumberOfStates, model.N);
            Assert.IsFalse(model.Normalized);
            for (int n = 0; n < _NumberOfStates; n++)
            {
                Assert.AreEqual(0.5, model.Pi[n]);
                Assert.IsNotNull(model.Emission[n]);
                Assert.IsInstanceOfType(model.Emission[n], typeof(Mixture<IMultivariateDistribution>));
                for (int i = 0; i < _NumberOfStates; i++)
                {
                    Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[n][i]);
                }
            }
        }        

        [TestMethod]
        public void Train_FTSESeriesAndErgodicModelAndLogNormalized_TrainedModel()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStates);
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);

            Assert.AreEqual(ModelType.Ergodic, model.Type);
            Assert.AreEqual(_NumberOfStates, model.Emission.Length);
        }

        [TestMethod]
        public void Train_FTSESeriesAndRightLeftAndLogNormalized_TrainedModel()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStatesRightLeft, Delta = _Delta });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStatesRightLeft, Delta);
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);

            Assert.AreEqual(ModelType.LeftRight, model.Type);
            Assert.AreEqual(_NumberOfStatesRightLeft, model.Emission.Length);
        }

        [TestMethod]
        public void Predict_FTSESeriesAndErgodicModel_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, _NumberOfIterations, _LikelihoodTolerance);

            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void Predict_FTSESeriesAndLeftRightModel_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStatesRightLeft, Delta = _Delta });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStatesRightLeft, Delta) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, _NumberOfIterations, _LikelihoodTolerance);

            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void Predict_FTSESeriesAndErgodicModelAnd20DaysTestSequence_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, 20, _NumberOfIterations, _LikelihoodTolerance);

            Assert.IsNotNull(result);
            Assert.AreEqual(20, result.Predicted.Length);
        }

        [TestMethod]
        public void Predict_FTSESeriesAndRightLeftModelAnd20DaysTestSequence_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStatesRightLeft, Delta = _Delta });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStatesRightLeft, Delta) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, 20, _NumberOfIterations, _LikelihoodTolerance);

            Assert.IsNotNull(result);
            Assert.AreEqual(20, result.Predicted.Length);
        }

        [TestMethod]
        public void EvaluatePrediction_FTSESeriesAndErgodicModelAnd20DaysTestSequence_PredictionEvaluated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, 20, _NumberOfIterations, _LikelihoodTolerance);
            var mape = model.EvaluatePrediction(result, test);

            Assert.IsNotNull(mape);
            Assert.AreEqual(4, mape.MeanAbsolutePercentError.Length);
        }

        [TestMethod]
        public void EvaluatePrediction_FTSESeriesAndRightLeftModelAnd20DaysTestSequence_PredictionEvaluated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStatesRightLeft, Delta = _Delta });//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStatesRightLeft, Delta) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, 20, _NumberOfIterations, _LikelihoodTolerance);
            var mape = model.EvaluatePrediction(result, test);

            Assert.IsNotNull(mape);
            Assert.AreEqual(4, mape.MeanAbsolutePercentError.Length);
        }

        [TestMethod]
        public void Predict_EURUSDSeriesAndErgodicModelAnd31DaysTestSequence_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.EURUSDFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.EURUSDFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, 31, _NumberOfIterations, _LikelihoodTolerance);

            Assert.IsNotNull(result);
            Assert.AreEqual(31, result.Predicted.Length);
        }
    }
}
