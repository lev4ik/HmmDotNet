using System;
using HmmDotNet.MachineLearning.Algorithms.Baum_Welch;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms
{
    [TestClass]
    public class TimeSensitiveWeightCalculatorTest
    {
        [TestMethod]
        public void Calculate_kIs5Tis5_EMA5isRo()
        {
            var result = TimeSensitiveWeightCalculator.Calculate(5, 5);

            Assert.AreEqual(result[4], 2d / (1d + 5d));
        }

        [TestMethod]
        public void Calculate_kIs3Tis5_EMAFrom1To3()
        {
            var k = 3;
            var T = 5;
            var r = 2d / (k + 1);

            var result = TimeSensitiveWeightCalculator.Calculate(k, T);

            Assert.AreEqual(result[0], Math.Pow(1 - r, T - k) * (1d / k));
            Assert.AreEqual(result[1], Math.Pow(1 - r, T - k) * (1d / k));
            Assert.AreEqual(result[2], Math.Pow(1 - r, T - k) * (1d / k));
        }

        [TestMethod]
        public void Calculate_kIs3Tis6_EMAFrom3To4()
        {
            var k = 3;
            var T = 6;
            var r = 2d / (k + 1);

            var result = TimeSensitiveWeightCalculator.Calculate(k, T);

            Assert.AreEqual(result[3], r * Math.Pow(1 - r, T - 3));
            Assert.AreEqual(result[4], r * Math.Pow(1 - r, T - 4));
        }
    
    }
}
