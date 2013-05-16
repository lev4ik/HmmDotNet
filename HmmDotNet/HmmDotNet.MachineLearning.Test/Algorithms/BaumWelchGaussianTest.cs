using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Test
{
    [TestClass]
    public class BaumWelchGaussianTest
    {
        private const int K = 3;
        private const int N = 41;
        private const int LikelihoodTolerance = 20;

        #region Private Variables

        private IList<IObservation> _observations;
        private double[] _startDistribution;
        private double[][] _tpm;
        private NormalDistribution[] _distributions;

        #endregion Private Variables

        [TestInitialize]
        public void Initialize()
        {
            var util = new TestDataUtils();

            var arr = util.GetSvcData(util.MSFTFilePath, new DateTime(2012, 1, 1), new DateTime(2012, 03, 1));

            _observations = new List<IObservation>();
            _startDistribution = new double[K];
            _tpm = new double[K][];
            _distributions = new NormalDistribution[K];

            _startDistribution[0] = 0.6d;
            _startDistribution[1] = 0.2d;
            _startDistribution[2] = 0.2d;

            for (var j = 0; j < K; j++)
            {
                _tpm[j] = (double[])_startDistribution.Clone();
            }

            var open = new double[N];
            var high = new double[N];
            var low = new double[N];

            for (var i = 0; i < N; i++)
            {
                _observations.Add(new Observation(new[] { arr[i][0] }, i.ToString()));
                open[i] = arr[i][0];
                high[i] = arr[i][1];
                low[i] = arr[i][2];
            }

            var d = new NormalDistribution(K);
            _distributions[0] = (NormalDistribution)d.Evaluate(open);
            _distributions[1] = (NormalDistribution)d.Evaluate(high);
            _distributions[2] = (NormalDistribution)d.Evaluate(low);
        }

        [TestMethod]
        public void TestBaumWelchGaussian()
        {
            var algo = new BaumWelchGaussianDistribution(_observations, HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<NormalDistribution>(){Pi = _startDistribution, TransitionProbabilityMatrix = _tpm, Emissions = _distributions}));//new HiddenMarkovModelGaussianDistribution(_startDistribution, _tpm, _distributions));

            var res = algo.Run(100, LikelihoodTolerance);

            Assert.AreEqual(1d, res.Pi.Sum());
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[0].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[1].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[2].Sum(), 5));
            var emissionValue = res.Emission[0].ProbabilityDensityFunction(new double[] { _observations[0].Value[0] });
            Assert.IsTrue(emissionValue > 0 && emissionValue < 1);
        }
    }
}
