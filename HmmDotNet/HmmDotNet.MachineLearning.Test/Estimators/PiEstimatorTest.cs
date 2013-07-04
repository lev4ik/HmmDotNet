using System;
using System.Collections.Generic;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions.Multivariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Estimators
{
    /// <summary>
    /// Summary description for PiEstimatorTest
    /// </summary>
    [TestClass]
    public class PiEstimatorTest
    {
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
        public void PiEstimator_ParameterPassed_PiEstimatorCreated()
        {
            var estimator = new PiEstimator();

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Estimate_ParametersPassed_PiCalculatedAndReturned()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Emissions = CreateEmissions(observations, numberOfStates) });
            model.Normalized = true;
            var observationsList = new List<IObservation>();

            for (var i = 0; i < observations.Length; i++)
            {
                observationsList.Add(new Observation(observations[i], i.ToString()));
            }
            var baseParameters = new BasicEstimationParameters<NormalDistribution> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<NormalDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<NormalDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            var @params = new AdvancedEstimationParameters<NormalDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = observationsList,
                Model = model,
                Normalized = model.Normalized
            };
            var gammaEstimator = new GammaEstimator<NormalDistribution>();

            var estimator = new PiEstimator();
            var parameters = new PiParameters
            {
                Gamma = gammaEstimator.Estimate(@params),
                N = model.N,
                Normalized = model.Normalized
            };

            var estimatedPi = estimator.Estimate(parameters);
            Assert.AreEqual(1d, Math.Round(estimatedPi[0] + estimatedPi[1], 5));
        }
    }
}
