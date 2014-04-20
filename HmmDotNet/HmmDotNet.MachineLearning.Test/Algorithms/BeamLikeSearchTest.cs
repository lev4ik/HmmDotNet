using System;
using HmmDotNet.Extentions;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms.Search;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms
{
    [TestClass]
    public class BeamLikeSearchTest
    {
        private readonly double[][] _trainingSet = new double[10][]
                {
                    new []{1353.36,1362.03,1343.35,1359.88},
                    new []{1355.41,1360.62,1348.05,1353.33},
                    new []{1374.64,1380.13,1352.5,1355.49},
                    new []{1380.03,1388.81,1371.39,1374.53},
                    new []{1379.86,1384.87,1377.19,1380},
                    new []{1377.55,1391.39,1373.03,1379.85},
                    new []{1394.53,1401.23,1377.51,1377.51},
                    new []{1428.27,1428.27,1388.14,1394.53},
                    new []{1417.26,1433.38,1417.26,1428.39},
                    new []{1414.02,1419.9,1408.13,1417.26}
                };

        private readonly double[] _node = new[] {1414.02, 1419.9, 1408.13, 1417.26};
        private const int _NumberOfComponents = 2;
        private const int _NumberOfIterations = 10;
        private const int _NumberOfStates = 2;
        private const int _LikelihoodTolerance = 20;

        [TestMethod]
        public void ExpandCurrentPopulation_OneNodeAndPopulationLengthOfTen_LengthTenPopulation()
        {
            var crf = new ChangeRatioFinder();

            var result = crf.GetMaximumChangeRatios(_trainingSet);

            var search = new BeamLikeSearch();
            var population = new double[1][]
                {
                    _node
                };

            var current = search.ExpandCurrentPopulation(population, 10, result.Down, result.Up);

            Assert.AreEqual(10, current.Length);
        }

        [TestMethod]
        public void ExpandCurrentPopulation_OneNodeAndPopulationLengthOfTen_LengthTenPopulationWithExactValues()
        {
            var crf = new ChangeRatioFinder();

            var result = crf.GetMaximumChangeRatios(_trainingSet);

            var search = new BeamLikeSearch();
            var population = new double[1][]
                {
                    _node
                };

            var current = search.ExpandCurrentPopulation(population, 10, result.Down, result.Up);

            Assert.AreEqual(1406.22, Math.Round(current[0][3], 2));
            Assert.AreEqual(1410.76, Math.Round(current[1][3], 2));
            Assert.AreEqual(1415.31, Math.Round(current[2][3], 2));
            Assert.AreEqual(1419.85, Math.Round(current[3][3], 2));
            Assert.AreEqual(1424.40, Math.Round(current[4][3], 2));
            Assert.AreEqual(1428.94, Math.Round(current[5][3], 2));
            Assert.AreEqual(1433.49, Math.Round(current[6][3], 2));
            Assert.AreEqual(1438.04, Math.Round(current[7][3], 2));
            Assert.AreEqual(1442.58, Math.Round(current[8][3], 2));
            Assert.AreEqual(1447.13, Math.Round(current[9][3], 2));
        }

        [TestMethod]
        public void Search_OneNodeAndKis3AndNis10_3Nodes()
        {
            var util = new TestDataUtils();
            var crf = new ChangeRatioFinder();
            var search = new BeamLikeSearch();

            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var ratios = crf.GetMaximumChangeRatios(series);           
            var population = new double[1][]
                {
                    series[series.Length]
                };
            var k = 3;
            var n = 10;
            var numberOfIterations = 10;

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);


           var result = search.Search(population, k, n, numberOfIterations, ratios, model);

            Assert.IsNull(result);
        }

    }
}
