using System;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class GammaEstimatorTest
    {
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
        }

        [TestMethod]
        public void GammaEstimator_ParametersAndNormalized_GammaEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new GammaEstimator<NormalDistribution>();

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Gamma_ErgodicAndLogNormalized_GammaCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);

            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = Helper.Convert(observations),
                Model = model,
                Normalized = model.Normalized
            };
            var estimator = new GammaEstimator<NormalDistribution>();

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < NumberOfStates; j++)
                {
                    Assert.IsTrue(estimator.Estimate(@params)[i][j] < 0, string.Format("Failed Gamma {0}", estimator.Estimate(@params)[i][j]));
                }
            }
        }

        [TestMethod]
        public void Gamma_ErgodicAndNotNormalized_GammaCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = false };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = Helper.Convert(observations),
                Model = model,
                Normalized = model.Normalized
            };
            var estimator = new GammaEstimator<NormalDistribution>();

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < NumberOfStates; j++)
                {
                    Assert.IsTrue(estimator.Estimate(@params)[i][j] > 0 && estimator.Estimate(@params)[i][j] < 1, string.Format("Failed Gamma {0}, [{1}][{2}]", estimator.Estimate(@params)[i][j], i, j));
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
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = Helper.Convert(observations),
                Model = model,
                Normalized = model.Normalized
            };
            var estimator = new GammaEstimator<NormalDistribution>();

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStatesRightLeft; j++)
                {
                    Assert.IsTrue(estimator.Estimate(@params)[i][j] <= 0 || double.IsNaN(estimator.Estimate(@params)[i][j]), string.Format("Failed Gamma [{1}][{2}] : {0}", estimator.Estimate(@params)[i][j], i, j));
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
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);

            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = Helper.Convert(observations),
                Model = model,
                Normalized = model.Normalized
            };
            var estimator = new GammaEstimator<NormalDistribution>();

            Assert.IsNotNull(estimator);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStatesRightLeft; j++)
                {
                    Assert.IsTrue(estimator.Estimate(@params)[i][j] >= 0 && estimator.Estimate(@params)[i][j] <= 1, string.Format("Failed Gamma [{1}][{2}] : {0}", estimator.Estimate(@params)[i][j], i, j));
                }
            }
        }

        [TestMethod]
        public void Gamma_ErgodicNotNormalized_EachEntryMatrixIsSummedToOne()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates) { LogNormalized = true };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = Helper.Convert(observations),
                Model = model,
                Normalized = model.Normalized
            };
            var estimator = new GammaEstimator<NormalDistribution>();
            for (int i = 0; i < observations.Length; i++)
            {
                Assert.AreEqual(1.0d, Math.Round(estimator.Estimate(@params)[i].Sum(), 5), string.Format("Failed Gamma Component [{1}] : {0}", estimator.Estimate(@params)[i], i));
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
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = Helper.Convert(observations),
                Model = model,
                Normalized = model.Normalized
            };
            var estimator = new GammaEstimator<NormalDistribution>();
            for (int i = 0; i < observations.Length; i++)
            {
                Assert.AreEqual(1.0d, Math.Round(estimator.Estimate(@params)[i].Sum(), 5), string.Format("Failed Gamma Component [{1}] : {0}", estimator.Estimate(@params)[i], i));
            }
        }
    }
}
