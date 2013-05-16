using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Algorithms
{
    [TestClass]
    public class ViterbiMultivariateTest
    {
        private const int K = 2;
        private const int N = 100;

        #region Private Variables

        private IList<IObservation> _observations;
        private double[] _startDistribution;
        private double[][] _tpm;
        private IMultivariateDistribution[] _distributions;
        private IList<IState> _states;

        #endregion Private Variables

        [TestInitialize]
        public void Initialize()
        {
            var util = new TestDataUtils();
            var msftData = util.GetSvcData(util.MSFTFilePath, new DateTime(2011, 10, 7), new DateTime(2012, 3, 1));
            var intlData = util.GetSvcData(util.INTLFilePath, new DateTime(2011, 10, 7), new DateTime(2012, 3, 1));

            //var msftData = SecurityDailyDataManager.GetRange("MSFT", new DateTime(2011, 10, 7), new DateTime(2012, 3, 1));
            //var nasdaqData = SecurityDailyDataManager.GetRange("^IXIC", new DateTime(2011, 10, 7), new DateTime(2012, 3, 1));
            _observations = new List<IObservation>();
            _startDistribution = new double[K];
            _tpm = new double[K][];
            _distributions = new IMultivariateDistribution[K];

            _states = new List<IState> {new State(0, "Stock1"), new State(1, "Stock2")};

            for (var j = 0; j < K; j++)
            {
                _startDistribution[j] = 1d / K;
            }
            for (var j = 0; j < K; j++)
            {
                _tpm[j] = (double[])_startDistribution.Clone();
            }

            var x = new double[N][];
            var y = new double[N][];

            for (var i = 0; i < N; i++)
            {
                //_observations.Add(new Observation(new [] { msftData[i].Open, msftData[i].Low, msftData[i].High, msftData[i].Close }, i.ToString(CultureInfo.InvariantCulture)));
                _observations.Add(new Observation(msftData[i], i.ToString(CultureInfo.InvariantCulture)));
                x[i] = msftData[i]; //new [] { msftData[i].Open, msftData[i].Low, msftData[i].High, msftData[i].Close };
                y[i] = intlData[i]; //new[] { intlData[i].Open, intlData[i].Low, intlData[i].High, intlData[i].Close };
            }

            var d = new NormalDistribution(4);
            var likelihood = 0.0d;
            _distributions[0] = (NormalDistribution)d.Evaluate(x, out likelihood);
            _distributions[1] = (NormalDistribution)d.Evaluate(y, out likelihood);
        }

        [TestMethod]
        public void Run_100LengthObservations2MultivariateDistribution_PathCount100()
        {
            var algo = new Viterbi(false);

            var path = algo.Run(_observations, _states, _startDistribution, _tpm, _distributions);

            Assert.AreEqual(path.Count, 100);
        }
    }
}
