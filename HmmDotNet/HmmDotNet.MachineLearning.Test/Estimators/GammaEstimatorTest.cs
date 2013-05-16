using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class GammaEstimatorTest
    {
        private AlphaEstimator<NormalDistribution> _alphaEstimator;
        private BetaEstimator<NormalDistribution> _betaEstimator;
        private const int NumberOfStates = 2;
        
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

        [TestInitialize]
        public void TestInitialized()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            _alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            _betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
        }

        [TestMethod]
        public void GammaEstimator_ParametersAndNormalized_GammaEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new GammaEstimator<NormalDistribution>(parameters, true);

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Gamma_ErgodicAndLogNormalized_GammaCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new GammaEstimator<NormalDistribution>(parameters, true);

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < NumberOfStates; j++)
                {
                    Assert.IsTrue(estimator.Gamma[i][j] < 0, string.Format("Failed Gamma {0}", estimator.Gamma[i][j]));
                }
            }
        }

        [TestMethod]
        public void Gamma_ErgodicAndNotNormalized_GammaCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = false };
            model.Normalized = false;
            _alphaEstimator.LogNormalized = false;
            _betaEstimator.LogNormalized = false;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new GammaEstimator<NormalDistribution>(parameters, false);

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < NumberOfStates; j++)
                {
                    Assert.IsTrue(estimator.Gamma[i][j] > 0 && estimator.Gamma[i][j] < 1, string.Format("Failed Gamma {0}, [{1}][{2}]", estimator.Gamma[i][j], i, j));
                }
            }            
        }

        [TestMethod]
        public void Gamma_RightLeftAndLogNormalized_GammaCalculated()
        {
            var delta = 3;
            var numberOfStatesRightLeft = 4;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, numberOfStatesRightLeft) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStatesRightLeft, delta, CreateEmissions(observations, numberOfStatesRightLeft)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), alphaEstimator.Alpha, betaEstimator.Beta);

            var estimator = new GammaEstimator<NormalDistribution>(parameters, true);

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStatesRightLeft; j++)
                {
                    Assert.IsTrue(estimator.Gamma[i][j] <= 0 || double.IsNaN(estimator.Gamma[i][j]), string.Format("Failed Gamma [{1}][{2}] : {0}", estimator.Gamma[i][j], i, j));
                }
            }
        }

        [TestMethod]
        public void Gamma_RightLeftAndNotNormalized_GammaCalculated()
        {
            var delta = 3;
            var numberOfStatesRightLeft = 4;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, numberOfStatesRightLeft) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStatesRightLeft, delta, CreateEmissions(observations, numberOfStatesRightLeft)) { LogNormalized = false };
            model.Normalized = false;
            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), alphaEstimator.Alpha, betaEstimator.Beta);

            var estimator = new GammaEstimator<NormalDistribution>(parameters, false);

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStatesRightLeft; j++)
                {
                    Assert.IsTrue(estimator.Gamma[i][j] >= 0 && estimator.Gamma[i][j] <= 1, string.Format("Failed Gamma [{1}][{2}] : {0}", estimator.Gamma[i][j], i, j));
                }
            }
        }

        [TestMethod]
        public void Gamma_ErgodicNotNormalized_EachEntryMatrixIsSummedToOne()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            _alphaEstimator.LogNormalized = false;
            _betaEstimator.LogNormalized = false;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new GammaEstimator<NormalDistribution>(parameters, false);
            for (int i = 0; i < observations.Length; i++)
            {
                Assert.AreEqual(1.0d, Math.Round(estimator.Gamma[i].Sum(), 5), string.Format("Failed Gamma Component [{1}] : {0}", estimator.Gamma[i], i));
            }
        }

        [TestMethod]
        public void Gamma_RightLeftNotNormalized_EachEntryMatrixIsSummedToOne()
        {
            var delta = 3;
            var numberOfStatesRightLeft = 4;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, numberOfStatesRightLeft) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStatesRightLeft, delta, CreateEmissions(observations, numberOfStatesRightLeft)) { LogNormalized = false };
            model.Normalized = false;
            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), alphaEstimator.Alpha, betaEstimator.Beta);

            var estimator = new GammaEstimator<NormalDistribution>(parameters, false);
            for (int i = 0; i < observations.Length; i++)
            {
                Assert.AreEqual(1.0d, Math.Round(estimator.Gamma[i].Sum(), 5), string.Format("Failed Gamma Component [{1}] : {0}", estimator.Gamma[i], i));
            }
        }
    }
}
