using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.Test
{
    [TestClass]
    public class BaumWelchMultivariateTest
    {
        private const int K = 3;
        private const int N = 21;
        private const int LikelihoodTolerance = 20;

        #region Private Variables

        private IList<IObservation> _observations;
        private double[] _startDistribution;
        private double[][] _tpm;
        private IMultivariateDistribution[] _emissions;

        #endregion Private Variables

        [TestInitialize]
        public void Initialize()
        {
            var util = new TestDataUtils();

            var up = util.GetSvcData(util.Sp500FilePath, new DateTime(2009, 9, 1), new DateTime(2010, 1, 1));
            var down = util.GetSvcData(util.Sp500FilePath, new DateTime(2010, 4, 20), new DateTime(2010, 7, 1));
            var treading = util.GetSvcData(util.Sp500FilePath, new DateTime(2011, 3, 1), new DateTime(2011, 6, 1));

            _startDistribution = new double[K];
            _tpm = new double[K][];
            _emissions = new IMultivariateDistribution[K];
            // Train Distribution
            var trainer = new NormalDistribution(4);
            double likelihood;
            _emissions[0] = (NormalDistribution)trainer.Evaluate(up, out likelihood);
            _emissions[1] = (NormalDistribution)trainer.Evaluate(down, out likelihood);
            _emissions[2] = (NormalDistribution)trainer.Evaluate(treading, out likelihood);
            // Train Start Distribution 
            for (var j = 0; j < K; j++)
            {
                _startDistribution[j] = 1d / K;
            }
            // Train Transition Probaboloty Matrix
            for (var j = 0; j < K; j++)
            {
                _tpm[j] = (double[])_startDistribution.Clone();
            }

            var arr = util.GetSvcData(util.Sp500FilePath, new DateTime(2012, 1, 1), new DateTime(2012, 2, 1));
            
            _observations = new List<IObservation>();
                                 
            for (var i = 0; i < N; i++)
            {
                _observations.Add(new Observation(arr[i], i.ToString()));
            }
        }

        [TestMethod]
        public void TestBaumWelchMultivariate()
        {
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>(){Pi = _startDistribution, TransitionProbabilityMatrix = _tpm, Emissions = _emissions});//new HiddenMarkovModelMultivariateGaussianDistribution(_startDistribution, _tpm, _emissions)
            model.Normalized = false;
            var algo = new BaumWelchMultivariateDistribution(_observations, model) {Normalized = false};
            var res = algo.Run(100, LikelihoodTolerance);

            Assert.AreEqual(1d, res.Pi.Sum());
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[0].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[1].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[2].Sum(), 5));
        }

    }
}
