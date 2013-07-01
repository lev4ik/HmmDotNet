using System;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using HmmDotNet.Statistics.Distributions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class MuEstimatorTest
    {
        private const int NumberOfStates = 2;
        private const int NumberOfStatesRightLeft = 4;

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
        public void MuEstimator_ModelAndObservations_MuEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) }); //new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new MuEstimator<NormalDistribution>(model, Helper.Convert(observations));

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Mu_MultivariateAndErgodicAndLogNormalized_MuCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var sequence = Helper.Convert(observations);
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = sequence, Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var parameters = new ParameterEstimations<NormalDistribution>(model, sequence, alpha, beta);

            var gammEstimator = new GammaEstimator<NormalDistribution>(parameters, true);
            var estimator = new MuEstimator<NormalDistribution>(model, Helper.Convert(observations));

            Assert.IsNotNull(estimator);
            var mu = estimator.MuMultivariate(gammEstimator.Gamma);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int j = 0; j < sequence[0].Dimention; j++)
                {
                    Assert.IsTrue(mu[i][j] > 0, string.Format("Failed Mu {0}", mu[i][j]));
                }
            }
        }

        [TestMethod]
        public void Mu_MultivariateAndErgodicAndNotNormalized_MuCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var sequence = Helper.Convert(observations);
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = false };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = sequence, Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var parameters = new ParameterEstimations<NormalDistribution>(model, sequence, alpha, beta);

            var gammEstimator = new GammaEstimator<NormalDistribution>(parameters, false);
            var estimator = new MuEstimator<NormalDistribution>(model, Helper.Convert(observations));

            Assert.IsNotNull(estimator);
            var mu = estimator.MuMultivariate(gammEstimator.Gamma);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int j = 0; j < sequence[0].Dimention; j++)
                {
                    Assert.IsTrue(mu[i][j] > 0, string.Format("Failed Mu {0}", mu[i][j]));
                }
            }
        }

        [TestMethod]
        public void Mu_MultivariateAndRightLeftAndLogNormalized_MuCalculated()
        {
            var delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var sequence = Helper.Convert(observations);
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, NumberOfStatesRightLeft) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStatesRightLeft, delta, CreateEmissions(observations, NumberOfStatesRightLeft)) { LogNormalized = true };
            model.Normalized = true;
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = sequence, Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var parameters = new ParameterEstimations<NormalDistribution>(model, sequence, alpha, beta);

            var gammEstimator = new GammaEstimator<NormalDistribution>(parameters, true);
            var estimator = new MuEstimator<NormalDistribution>(model, Helper.Convert(observations));

            Assert.IsNotNull(estimator);
            var mu = estimator.MuMultivariate(gammEstimator.Gamma);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int j = 0; j < sequence[0].Dimention; j++)
                {
                    Assert.IsTrue(mu[i][j] > 0, string.Format("Failed Mu {0}", mu[i][j]));
                }
            }
        }

        [TestMethod]
        public void Mu_MultivariateAndRightLeftAndNotNormalized_MuCalculated()
        {
            var delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var sequence = Helper.Convert(observations);
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, NumberOfStatesRightLeft) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStatesRightLeft, delta, CreateEmissions(observations, NumberOfStatesRightLeft)) { LogNormalized = false };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = sequence, Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var parameters = new ParameterEstimations<NormalDistribution>(model, sequence, alpha, beta);

            var gammEstimator = new GammaEstimator<NormalDistribution>(parameters, false);
            var estimator = new MuEstimator<NormalDistribution>(model, Helper.Convert(observations));

            Assert.IsNotNull(estimator);
            var mu = estimator.MuMultivariate(gammEstimator.Gamma);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int j = 0; j < sequence[0].Dimention; j++)
                {
                    Assert.IsTrue(mu[i][j] > 0, string.Format("Failed Mu {0}", mu[i][j]));
                }
            }
        }
    }
}
