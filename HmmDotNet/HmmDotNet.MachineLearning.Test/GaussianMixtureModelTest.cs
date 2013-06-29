using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.GaussianMixtureModels;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.Test
{
    [TestClass]
    public class GaussianMixtureModelTest
    {
        #region Members

        private double[] _coefficients = new double[K];
        private NormalDistribution[] _distributions = new NormalDistribution[K];
        private double[][] _observations = new double[N][];
        private double[][] _observed = new double[20][];
        private double[] _weights = new double[N];
        private const int LikelihoodTolerance = 20;

        #endregion Members

        [TestInitialize]
        public void Initialize()
        {
            var util = new TestDataUtils();
            
            // Create 2 Distributions
            // Stock + Ema7, Stock + Ema28 => Data : Stock + Index
            _observations = util.GetSvcData(util.MSFTFilePath, new DateTime(2011, 10, 7), new DateTime(2012, 3, 1));//SecurityDailyDataManager.GetRange("MSFT", new DateTime(2011, 10, 7), new DateTime(2012, 3, 1));
            _observed = util.GetSvcData(util.MSFTFilePath, new DateTime(2012, 3, 2), new DateTime(2012, 3, 29));
            var trainData1 = util.GetSvcData(util.MSFTFilePath, new DateTime(2010, 1, 1), new DateTime(2011, 1, 1));
            var trainData2 = util.GetSvcData(util.MSFTFilePath, new DateTime(2009, 1, 1), new DateTime(2010, 1, 1));          
          
            var d0 = new NormalDistribution(4);
            var likelihood0 = 0.0d;
            _distributions[0] = (NormalDistribution)d0.Evaluate(trainData1, out likelihood0);

            var d1 = new NormalDistribution(4);
            var likelihood1 = 0.0d;
            _distributions[1] = (NormalDistribution)d1.Evaluate(trainData2, out likelihood1);

            for (var k = 0; k < K; k++)
            {
                _coefficients[k] = 1d / K;
            }
        }

        /// <summary>
        ///     Number of components
        /// </summary>
        public const int K = 2;

        /// <summary>
        ///     Number of observation points 
        /// </summary>
        public const int N = 100;

        [TestMethod]
        public void Train_CoefficientsAndDistributions_TrainedModel()
        {
            var model = new GaussianMixtureModel(_coefficients, _distributions);
            model.Train(_observations, 100, LikelihoodTolerance);
            Assert.AreEqual(Math.Round(model.Likelihood, 4), -95.5274);
            Assert.AreEqual(model.Mixture.Dimension, 4);
        }

        [TestMethod]
        public void Predict_CoefficientsAndDistributions_Prediction()
        {
            var model = new GaussianMixtureModel(_coefficients, _distributions);
            model.Train(_observations, 100, LikelihoodTolerance);
            var prediction = model.Predict(_observations, null);
            Assert.AreEqual(prediction.Predicted[0][0], 32.480000000000004);
            Assert.AreEqual(prediction.Predicted[0][1], 32.94);
            Assert.AreEqual(prediction.Predicted[0][2], 32.24);
            Assert.AreEqual(prediction.Predicted[0][3], 33.03);
        }
    }
}
