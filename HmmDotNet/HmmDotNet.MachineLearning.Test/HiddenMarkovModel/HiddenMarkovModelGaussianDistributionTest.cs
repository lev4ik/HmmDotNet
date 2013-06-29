using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Test
{
    /// <summary>
    /// Summary description for HMMTest
    /// </summary>
    [TestClass]
    public class HiddenMarkovModelGaussianDistributionTest
    {
        private double[] _initialPi;
        private double[][] _initialTransitionProbabilityMatrix;
        private NormalDistribution[] _initialEmission;
        private TestDataUtils _dataUtil;
        private double[] _observationClose;
        private double[] _observationLow;
        private double[] _observationHigh;
        private double[] _observationOpen;
        private double[] _mean;

        [TestInitialize]
        public void InitiateForTraining()
        {
            var n = 4; // number of states
            _initialPi = new double[n];
            _initialTransitionProbabilityMatrix = new double[n][];
            _initialEmission = new NormalDistribution[n];
            _dataUtil = new TestDataUtils();
            var data = _dataUtil.GetSvcData(_dataUtil.Sp500FilePath, new DateTime(2012, 1, 1), new DateTime(2012, 6, 1));
            _observationClose = new double[data.Length];
            _observationOpen = new double[data.Length];
            _observationHigh = new double[data.Length];
            _observationLow = new double[data.Length];

            for (var k = 0; k < data.Length; k++)
            {
                _observationOpen[k] = data[k][0];
                _observationLow[k] = data[k][1];
                _observationHigh[k] = data[k][2];
                _observationClose[k] = data[k][3];
            }

            _mean = new double[n];
            _mean[0] = _observationOpen.Mean();
            _mean[1] = _observationLow.Mean();
            _mean[2] = _observationHigh.Mean();
            _mean[3] = _observationClose.Mean();

            for (var i = 0; i < n; i++)
            {
                _initialPi[i] = 1d / n;
                _initialEmission[i] = new NormalDistribution(_mean[i], _mean[i] / 2);
                _initialTransitionProbabilityMatrix[i] = new double[n];
                for (var j = 0; j < n; j++)
                {
                    if (i != j)
                    {
                        _initialTransitionProbabilityMatrix[i][j] = 1d / (n - 1);
                    }
                }
            }
        }
    }
}
