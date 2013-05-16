using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class KsiEstimatorTest
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
        public void KsiEstimator_Parameters_KsiEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new KsiEstimator<NormalDistribution>(parameters, true);

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Ksi_ErgodicAndLogNormalized_KsiCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new KsiEstimator<NormalDistribution>(parameters, true);

            Assert.IsNotNull(estimator);
            for (int t = 0; t < observations.Length - 1; t++)
            {
                for (int i = 0; i < NumberOfStates; i++)
                {
                    for (int j = 0; j < NumberOfStates; j++)
                    {
                        Assert.IsTrue(estimator.Ksi[t][i, j] < 0, string.Format("Failed Ksi {0}", estimator.Ksi[t][i, j]));
                    }                    
                }
            }
        }

        [TestMethod]
        public void Ksi_ErgodicAndNotNormalized_KsiCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            _alphaEstimator.LogNormalized = false;
            _betaEstimator.LogNormalized = false;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new KsiEstimator<NormalDistribution>(parameters, false);

            Assert.IsNotNull(estimator);
            for (int t = 0; t < observations.Length - 1; t++)
            {
                for (int i = 0; i < NumberOfStates; i++)
                {
                    for (int j = 0; j < NumberOfStates; j++)
                    {
                        Assert.IsTrue(estimator.Ksi[t][i, j] > 0 && estimator.Ksi[t][i, j] < 1, string.Format("Failed Ksi {0}", estimator.Ksi[t][i, j]));
                    }
                }
            }
        }

        [TestMethod]
        public void Ksi_RightLeftAndLogNormalized_KsiCalculated()
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

            var estimator = new KsiEstimator<NormalDistribution>(parameters, true);

            Assert.IsNotNull(estimator);
            for (int t = 0; t < observations.Length - 1; t++)
            {
                for (int i = 0; i < numberOfStatesRightLeft; i++)
                {
                    for (int j = 0; j < numberOfStatesRightLeft; j++)
                    {
                        Assert.IsTrue(estimator.Ksi[t][i, j] < 0 || double.IsNaN(estimator.Ksi[t][i, j]), string.Format("Failed Ksi [{1}][{2},{3}] : {0}", estimator.Ksi[t][i, j], t, i, j));
                    }
                }
            }
        }

        [TestMethod]
        public void Ksi_RightLeftAndNotNormalized_KsiCalculated()
        {
            var delta = 3;
            var numberOfStatesRightLeft = 4;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, numberOfStatesRightLeft) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStatesRightLeft, delta, CreateEmissions(observations, numberOfStatesRightLeft)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);

            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), alphaEstimator.Alpha, betaEstimator.Beta);

            var estimator = new KsiEstimator<NormalDistribution>(parameters, false);

            Assert.IsNotNull(estimator);
            for (int t = 0; t < observations.Length - 1; t++)
            {
                for (int i = 0; i < numberOfStatesRightLeft; i++)
                {
                    for (int j = 0; j < numberOfStatesRightLeft; j++)
                    {
                        Assert.IsTrue(estimator.Ksi[t][i, j] >= 0 && estimator.Ksi[t][i, j] < 1, string.Format("Failed Ksi [{1}][{2},{3}]:{0}", estimator.Ksi[t][i, j], t, i, j));
                    }
                }
            }
        }

        [TestMethod]
        public void Ksi_ErgodicNotNormalized_EachEntryMatrixIsSummedToOne()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            _alphaEstimator.LogNormalized = false;
            _betaEstimator.LogNormalized = false;
            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), _alphaEstimator.Alpha, _betaEstimator.Beta);

            var estimator = new KsiEstimator<NormalDistribution>(parameters, false);
            for (int t = 0; t < observations.Length - 1; t++)
            {
                Assert.AreEqual(1.0d, Math.Round(estimator.Ksi[t].Sum(), 5), string.Format("Failed Ksi [{1}] :{0}", new Matrix(estimator.Ksi[t]), t));
            }
        }

        [TestMethod]
        public void Ksi_RightLeftNotNormalized_EachEntryMatrixIsSummedToOne()
        {
            var delta = 3;
            var numberOfStatesRightLeft = 4;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, numberOfStatesRightLeft) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStatesRightLeft, delta, CreateEmissions(observations, numberOfStatesRightLeft)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);

            var parameters = new ParameterEstimations<NormalDistribution>(model, Helper.Convert(observations), alphaEstimator.Alpha, betaEstimator.Beta);

            var estimator = new KsiEstimator<NormalDistribution>(parameters, false);
            for (int t = 0; t < observations.Length - 1; t++)
            {
                Assert.AreEqual(1.0d, Math.Round(estimator.Ksi[t].Sum(), 5), string.Format("Failed Ksi [{1}] :{0}", new Matrix(estimator.Ksi[t]), t));
            }
        }

    }
}
