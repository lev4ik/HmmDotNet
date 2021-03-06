﻿using System;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class SigmaEstimatorTest
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
        public void SigmaEstimator_ModelAndObservations_SigmaEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(NumberOfStates, CreateEmissions(observations, NumberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            var estimator = new SigmaMultivariateEstimator<NormalDistribution>();

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Sigma_ErgodicAndObservationAndLogNormalized_SigmaCalculated()
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

            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = sequence,
                Model = model
            };
            var gammaEstimator = new GammaEstimator<NormalDistribution>();
            var muEstimator = new MuMultivariateEstimator<NormalDistribution>();
            var estimator = new SigmaMultivariateEstimator<NormalDistribution>();
            var muParams = new MuEstimationParameters<NormalDistribution>
            {
                Gamma = gammaEstimator.Estimate(@params),
                Model = model,
                Normalized = model.Normalized,
                Observations = Helper.Convert(observations)
            };
            Assert.IsNotNull(estimator);
            var sigma = estimator.Estimate(new SigmaEstimationParameters<NormalDistribution, double[][]>(muParams) { Mean = muEstimator.Estimate(muParams) });

            for (int n = 0; n < NumberOfStates; n++)
            {
                for (int i = 0; i < sequence[0].Dimention; i++)
                {
                    for (int j = 0; j < sequence[0].Dimention; j++)
                    {
                        Assert.IsTrue(sigma[n][i, j] > 0, string.Format("Failed Sigma {0}", sigma[n][i, j]));
                    }                   
                }
            }
        }
        
        [TestMethod]
        public void Sigma_ErgodicAndObservationAndNotNormalized_SigmaCalculated()
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
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = sequence,
                Model = model
            };

            var gammaEstimator = new GammaEstimator<NormalDistribution>();
            var muEstimator = new MuMultivariateEstimator<NormalDistribution>();
            var estimator = new SigmaMultivariateEstimator<NormalDistribution>();
            var muParams = new MuEstimationParameters<NormalDistribution>
            {
                Gamma = gammaEstimator.Estimate(@params),
                Model = model,
                Normalized = model.Normalized,
                Observations = Helper.Convert(observations)
            };
            Assert.IsNotNull(estimator);
            var sigma = estimator.Estimate(new SigmaEstimationParameters<NormalDistribution, double[][]>(muParams) { Mean = muEstimator.Estimate(muParams) });

            for (int n = 0; n < NumberOfStates; n++)
            {
                for (int i = 0; i < sequence[0].Dimention; i++)
                {
                    for (int j = 0; j < sequence[0].Dimention; j++)
                    {
                        Assert.IsTrue(sigma[n][i, j] > 0, string.Format("Failed Sigma {0}", sigma[n][i, j]));
                    }
                }
            }
        }

        [TestMethod]
        public void Sigma_LeftRightAndObservationAndLogNormalized_SigmaCalculated()
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
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = sequence,
                Model = model
            };
            var gammaEstimator = new GammaEstimator<NormalDistribution>();
            var muEstimator = new MuMultivariateEstimator<NormalDistribution>();
            var estimator = new SigmaMultivariateEstimator<NormalDistribution>();
            var muParams = new MuEstimationParameters<NormalDistribution>
            {
                Gamma = gammaEstimator.Estimate(@params),
                Model = model,
                Normalized = model.Normalized,
                Observations = Helper.Convert(observations)
            };
            Assert.IsNotNull(estimator);
            var sigma = estimator.Estimate(new SigmaEstimationParameters<NormalDistribution, double[][]>(muParams) { Mean = muEstimator.Estimate(muParams) });

            for (int n = 0; n < NumberOfStatesRightLeft; n++)
            {
                for (int i = 0; i < sequence[0].Dimention; i++)
                {
                    for (int j = 0; j < sequence[0].Dimention; j++)
                    {
                        Assert.IsTrue(sigma[n][i, j] > 0, string.Format("Failed Sigma {0}", sigma[n][i, j]));
                    }
                }
            }
        }

        [TestMethod]
        public void Sigma_LeftRightAndObservationAndNotNormalized_SigmaCalculated()
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
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = sequence,
                Model = model
            };
            var gammaEstimator = new GammaEstimator<NormalDistribution>();
            var muEstimator = new MuMultivariateEstimator<NormalDistribution>();
            var estimator = new SigmaMultivariateEstimator<NormalDistribution>();
            var muParams = new MuEstimationParameters<NormalDistribution>
            {
                Gamma = gammaEstimator.Estimate(@params),
                Model = model,
                Normalized = model.Normalized,
                Observations = Helper.Convert(observations)
            };
            Assert.IsNotNull(estimator);
            var sigma = estimator.Estimate(new SigmaEstimationParameters<NormalDistribution, double[][]>(muParams) { Mean = muEstimator.Estimate(muParams) });

            for (int n = 0; n < NumberOfStatesRightLeft; n++)
            {
                for (int i = 0; i < sequence[0].Dimention; i++)
                {
                    for (int j = 0; j < sequence[0].Dimention; j++)
                    {
                        Assert.IsTrue(sigma[n][i, j] > 0, string.Format("Failed Sigma {0}", sigma[n][i, j]));
                    }
                }
            }
        }
    }
}
