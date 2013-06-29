using System;
using System.Collections.Generic;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Estimators
{
    [TestClass]
    public class TransitionProbabilityEstimatorTest
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
        public void TransitionProbabilityEstimator_ParameterPassed_TransitionProbabilityEstimatorCreated()
        {
            var estimator = new TransitionProbabilityEstimator<IDistribution>();

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Estimate_KsiGammaParameters_TransitionProbabilityMatrixCalculatedAndReturned()
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

            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), model.Normalized);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), model.Normalized);
            var estimationParameters = new ParameterEstimations<NormalDistribution>(model, observationsList, alphaEstimator.Alpha, betaEstimator.Beta);

            var gammaEstimator = new GammaEstimator<NormalDistribution>(estimationParameters, model.Normalized);
            var ksiEstimator = new KsiEstimator<NormalDistribution>(estimationParameters, model.Normalized);

            var estimator = new TransitionProbabilityEstimator<NormalDistribution>();
            var parameters = new KsiGammaTransitionProbabilityMatrixParameters<NormalDistribution>
                {
                    Model = model,
                    Ksi = ksiEstimator.Ksi,
                    Gamma = gammaEstimator.Gamma,
                    T = observations.Length,
                    Normalized = model.Normalized
                };

            var estimatedTransitionProbabilityMatrix = estimator.Estimate(parameters);
            Assert.AreEqual(1d, Math.Round(estimatedTransitionProbabilityMatrix[0][0] + estimatedTransitionProbabilityMatrix[0][1], 5));
            Assert.AreEqual(1d, Math.Round(estimatedTransitionProbabilityMatrix[1][0] + estimatedTransitionProbabilityMatrix[1][1], 5));
        }

        [TestMethod]
        public void Estimate_AlphaBetaParameters_TransitionProbabilityMatrixCalculatedAndReturned()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Emissions = CreateEmissions(observations, numberOfStates) });
            model.Normalized = true;

            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), model.Normalized);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), model.Normalized);
            var weights = new double[observations.Length];

            var estimator = new TransitionProbabilityEstimator<NormalDistribution>();
            var parameters = new AlphaBetaTransitionProbabiltyMatrixParameters<NormalDistribution>
                {
                    Alpha = alphaEstimator.Alpha,
                    Beta = betaEstimator.Beta,
                    Model = model,
                    Observations = observations,
                    Normalized = model.Normalized,
                    Weights = weights
                };

            var estimatedTransitionProbabilityMatrix = estimator.Estimate(parameters);
            Assert.AreEqual(1d, Math.Round(estimatedTransitionProbabilityMatrix[0][0] + estimatedTransitionProbabilityMatrix[0][1], 5));
            Assert.AreEqual(1d, Math.Round(estimatedTransitionProbabilityMatrix[1][0] + estimatedTransitionProbabilityMatrix[1][1], 5));
        }

        [TestMethod]
        public void Compare_AlphaBetaAndKsiGammaCalculation_EqualTransitionProbabilityMatrixes()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Emissions = CreateEmissions(observations, numberOfStates) });
            model.Normalized = true;
            var observationsList = new List<IObservation>();
            var weights = new double[observations.Length];
            for (var i = 0; i < observations.Length; i++)
            {
                observationsList.Add(new Observation(observations[i], i.ToString()));
            }

            var alphaEstimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), model.Normalized);
            var betaEstimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), model.Normalized);
            var estimationParameters = new ParameterEstimations<NormalDistribution>(model, observationsList, alphaEstimator.Alpha, betaEstimator.Beta);

            var gammaEstimator = new GammaEstimator<NormalDistribution>(estimationParameters, model.Normalized);
            var ksiEstimator = new KsiEstimator<NormalDistribution>(estimationParameters, model.Normalized);

            var estimatorKsiGamma = new TransitionProbabilityEstimator<NormalDistribution>();
            var parametersKsiGamma = new KsiGammaTransitionProbabilityMatrixParameters<NormalDistribution>
            {
                Model = model,
                Ksi = ksiEstimator.Ksi,
                Gamma = gammaEstimator.Gamma,
                T = observations.Length,
                Normalized = model.Normalized
            };
            var estimatorAlphaBeta = new TransitionProbabilityEstimator<NormalDistribution>();
            var parametersAlphaBeta = new AlphaBetaTransitionProbabiltyMatrixParameters<NormalDistribution>
            {
                Alpha = alphaEstimator.Alpha,
                Beta = betaEstimator.Beta,
                Model = model,
                Observations = observations,
                Normalized = model.Normalized,
                Weights = weights
            };

            var estimatedTransitionProbabilityMatrixKsiGamma = estimatorKsiGamma.Estimate(parametersKsiGamma);
            var estimatedTransitionProbabilityMatrixAlphaBeta = estimatorAlphaBeta.Estimate(parametersAlphaBeta);

            Assert.AreEqual(Math.Round(estimatedTransitionProbabilityMatrixKsiGamma[0][0], 5), Math.Round(estimatedTransitionProbabilityMatrixAlphaBeta[0][0], 5));
            Assert.AreEqual(Math.Round(estimatedTransitionProbabilityMatrixKsiGamma[0][1], 5), Math.Round(estimatedTransitionProbabilityMatrixAlphaBeta[0][1], 5));
            Assert.AreEqual(Math.Round(estimatedTransitionProbabilityMatrixKsiGamma[1][0], 5), Math.Round(estimatedTransitionProbabilityMatrixAlphaBeta[1][0], 5));
            Assert.AreEqual(Math.Round(estimatedTransitionProbabilityMatrixKsiGamma[1][1], 5), Math.Round(estimatedTransitionProbabilityMatrixAlphaBeta[1][1], 5));
        }
    }
}
