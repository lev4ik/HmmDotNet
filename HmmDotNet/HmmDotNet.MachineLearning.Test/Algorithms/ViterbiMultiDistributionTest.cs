using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Algorithms
{
    [TestClass]
    public class ViterbiMultiDistributionTest
    {
        private int _NumberOfComponents = 4; 
        private int _NumberOfStates = 4;
        private int _NumberOfDistributionsInState = 2;
        private double[][] _distributionWeights;
        private IDistribution[][] _distributions;

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


        private void InitializeWeightsAndDistributionsMixture()
        {
            _distributionWeights = new double[_NumberOfStates][];
            for (var i = 0; i < _NumberOfStates; i++)
            {
                _distributionWeights[i] = new double[_NumberOfDistributionsInState];
                _distributionWeights[i][0] = 0.5;
                _distributionWeights[i][1] = 0.5;
            }
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var distributions = CreateEmissions(series, _NumberOfStates * _NumberOfComponents * _NumberOfDistributionsInState);
            _distributions = new Mixture<IMultivariateDistribution>[_NumberOfStates][];
            var index = 0;
            for (int i = 0; i < _NumberOfStates; i++)
            {
                _distributions[i] = new Mixture<IMultivariateDistribution>[_NumberOfDistributionsInState];
                for (int k = 0; k < _NumberOfDistributionsInState; k++)
                {
                    _distributions[i][k] = new Mixture<IMultivariateDistribution>(_NumberOfComponents, series[0].Length);
                    for (int j = 0; j < _NumberOfComponents; j++)
                    {
                        ((Mixture<IMultivariateDistribution>)_distributions[i][k]).Components[j] = distributions[index];
                        index++;
                    }
                }
            }
        }

        private void InitializeWeightsAndDistributionsMultivariate()
        {
            _distributionWeights = new double[_NumberOfStates][];
            for (var i = 0; i < _NumberOfStates; i++)
            {
                _distributionWeights[i] = new double[_NumberOfDistributionsInState];
                _distributionWeights[i][0] = 0.5;
                _distributionWeights[i][1] = 0.5;
            }
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var distributions = CreateEmissions(series, _NumberOfStates * _NumberOfDistributionsInState);
            _distributions = new IMultivariateDistribution[_NumberOfStates][];
            var index = 0;
            for (int i = 0; i < _NumberOfStates; i++)
            {
                _distributions[i] = new IMultivariateDistribution[_NumberOfDistributionsInState];
                for (int k = 0; k < _NumberOfDistributionsInState; k++)
                {
                    _distributions[i][k] = distributions[index];
                    index++;
                }
            }
        }

        [TestMethod]
        public void Run_FTSEObservationAndMixtureArray_MPPCalculated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var alg = new ViterbiMultiDistribution(true);
            var states = new List<IState>() {new State(0, "0"),new State(1, "1"),new State(2, "2"),new State(3, "3")};
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            InitializeWeightsAndDistributionsMixture();

            var mpp = alg.Run(Helper.Convert(series), states, model.Pi, model.TransitionProbabilityMatrix, _distributionWeights, _distributions);

            Assert.AreEqual(series.Length, mpp.Count);
        }

        [TestMethod]
        public void Run_FTSEObservationAndMultivariateGaussianArray_MPPCalculated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var alg = new ViterbiMultiDistribution(true);
            var states = new List<IState>() { new State(0, "0"), new State(1, "1"), new State(2, "2"), new State(3, "3") };
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            InitializeWeightsAndDistributionsMultivariate();

            var mpp = alg.Run(Helper.Convert(series), states, model.Pi, model.TransitionProbabilityMatrix, _distributionWeights, _distributions);

            Assert.AreEqual(series.Length, mpp.Count);            
        }
    }
}
