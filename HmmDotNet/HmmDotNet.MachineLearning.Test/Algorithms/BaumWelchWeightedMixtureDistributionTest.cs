using System;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms.Baum_Welch;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms
{
    /// <summary>
    /// Summary description for BaumWelchWeightedMixtureDistributionTest
    /// </summary>
    [TestClass]
    public class BaumWelchWeightedMixtureDistributionTest
    {
        private const int LikelihoodTolerance = 20;
        private const int MaxIterationNumber = 100;
        private const int NumberOfComponents = 4;
        private const int NumberOfStates = 3;

        private double[] _pi = new double[NumberOfStates];
        private double[][] _tpm = new double[NumberOfStates][];
        private Mixture<IMultivariateDistribution>[] _emission = new Mixture<IMultivariateDistribution>[NumberOfStates];
        
        private void Initialize(double[][] observations)
        {
            var algo = new KMeans();

            for (var j = 0; j < NumberOfStates; j++)
            {
                _pi[j] = 1d / NumberOfStates;
            }

            for (var j = 0; j < NumberOfStates; j++)
            {
                _tpm[j] = (double[])_pi.Clone();
            }

            var k = _pi.Length * NumberOfComponents;
            var dimentions = observations[0].Length;
            algo.CreateClusters(observations, k, KMeans.KMeansDefaultIterations, (k > 3) ? InitialClusterSelectionMethod.Furthest : InitialClusterSelectionMethod.Random);
            _emission = new Mixture<IMultivariateDistribution>[_pi.Length];

            for (int i = 0; i < _pi.Length; i++)
            {
                _emission[i] = new Mixture<IMultivariateDistribution>(NumberOfComponents, dimentions);
                for (int j = 0; j < NumberOfComponents; j++)
                {
                    var mean = algo.ClusterCenters[j + NumberOfComponents * i];
                    var covariance = algo.ClusterCovariances[j + NumberOfComponents * i];

                    _emission[i].Components[j] = new NormalDistribution(mean, covariance);
                }
            }
        }

        [TestMethod]
        public void Run_TrainingSetAndModel_TrainedModel()
        {
            var util = new TestDataUtils();
            var observationArray = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var observations = Helper.Convert(observationArray);
            var observationsWeight = Array.ConvertAll(TimeSensitiveWeightCalculator.Calculate(14, observations.Count), x => (decimal)x);

            Initialize(observationArray);
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { Pi = _pi, TransitionProbabilityMatrix = _tpm, Emissions = _emission});
            
            model.Normalized = true;

            var algo = new BaumWelchWeightedMixtureDistribution(observations, observationsWeight, model);
            var trainedModelState = algo.Run(MaxIterationNumber, LikelihoodTolerance);

            Assert.AreEqual(1d, Math.Round(trainedModelState.Pi.Sum(), 5));
            Assert.AreEqual(1d, Math.Round(trainedModelState.TransitionProbabilityMatrix[0].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(trainedModelState.TransitionProbabilityMatrix[1].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(trainedModelState.TransitionProbabilityMatrix[2].Sum(), 5));
        }
    }
}
