using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Distance;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class DistanceTest
    {
        [TestMethod]
        public void Calculate_MeanVectorAndCovarianceMatrixAndVector_MahalanobisDistance()
        {
            var m = new double[] {500, 500};
            var x = new double[] {410, 400};
            var sigma = new double[2,2];
            sigma[0, 0] = 6291.55737;
            sigma[0, 1] = 3754.32851;
            sigma[1, 0] = 3754.32851;
            sigma[1, 1] = 6280.77066;
            var matrix = new Matrix(sigma);
            var sigmaInverse = matrix.Inverse();
            sigmaInverse[0, 0] = Math.Round(sigmaInverse[0, 0], 5);
            sigmaInverse[0, 1] = Math.Round(sigmaInverse[0, 1], 5);
            sigmaInverse[1, 0] = Math.Round(sigmaInverse[1, 0], 5);
            sigmaInverse[1, 1] = Math.Round(sigmaInverse[1, 1], 5);

            var d = Mahalanobis.Calculate(x, m, sigmaInverse);
            
            Assert.AreEqual(1.825, Math.Round(d, 3));
        }

        [TestMethod]
        public void Calculate_TwoVectors_EuclideanDistance()
        {
            var x = new double[] { 500, 500 };
            var y = new double[] { 410, 400 };

            var d = Euclidean.Calculate(x, y);

            Assert.AreEqual(134.53624, Math.Round(d, 6));
        }
    }
}
