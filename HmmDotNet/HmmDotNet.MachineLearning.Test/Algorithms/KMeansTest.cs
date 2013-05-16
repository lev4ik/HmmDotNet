using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms.Clustering;

namespace HmmDotNet.Logic.Test.MachineLearning
{
    [TestClass]
    public class KMeansTest
    {
        [TestMethod]
        public void CreateClusters_FTSEObservations_2Clusters()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var algo = new KMeans();
            algo.CreateClusters(series, 2, 200, InitialClusterSelectionMethod.Furthest);

            Assert.AreEqual(2, algo.Clusters.Count);
            for (int i = 0; i < algo.Clusters.Count; i++)
            {
                Assert.IsTrue(algo.Clusters[i].Length > 0);
            }
        }

        [TestMethod]
        public void CreateClusters_FTSEObservations_3Clusters()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var algo = new KMeans();
            algo.CreateClusters(series, 3, 200, InitialClusterSelectionMethod.Furthest);

            Assert.AreEqual(3, algo.Clusters.Count);
            for (int i = 0; i < algo.Clusters.Count; i++)
            {
                Assert.IsTrue(algo.Clusters[i].Length > 0);
            }
        }

        [TestMethod]
        public void CreateClusters_FTSEObservations_4Clusters()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var algo = new KMeans();
            algo.CreateClusters(series, 4, 200, InitialClusterSelectionMethod.Random);

            Assert.AreEqual(4, algo.Clusters.Count);
            for (int i = 0; i < algo.Clusters.Count; i++)
            {
                Assert.IsTrue(algo.Clusters[i].Length > 0);
            }
        }
    }
}
