using System;
using System.Collections.Generic;
using System.Linq;
using HmmDotNet.TechnicalAnalysis.MovingAverage;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.TechnicalAnalysis.Test.MovingAverage
{
    [TestClass]
    public class WeightedMovingAverageTest
    {
        [TestMethod]
        public void Calculate_PointsIsNull_ArgumentExceptionThrown()
        {
            var calc = new WeightedMovingAverage();

            try
            {
                calc.Calculate(null, null);
            }
            catch (Exception ex)
            {
                Assert.IsInstanceOfType(ex, typeof(ArgumentException));
            }
        }

        [TestMethod]
        public void Calculate_WeightIsNull_ArgumentExceptionThrown()
        {
            var calc = new WeightedMovingAverage();
            var points = new List<double>();

            try
            {
                calc.Calculate(points, null);
            }
            catch (Exception ex)
            {
                Assert.IsInstanceOfType(ex, typeof(ArgumentException));
            }
        }

        [TestMethod]
        public void Calculate_PointsAndWeightsNotEqualInLength_ArgumentExceptionThrown()
        {
            var calc = new WeightedMovingAverage();
            var points = new List<double>() {0.1,0.2};
            var weights = new List<double>() {0.1};

            try
            {
                calc.Calculate(points, weights);
            }
            catch (Exception ex)
            {
                Assert.IsInstanceOfType(ex, typeof(ArgumentException));
            }
        }

        [TestMethod]
        public void Calculate_10PointsAllOnesWithSameWeights_1()
        {
            var calc = new WeightedMovingAverage();
            var points = Enumerable.Repeat(1d, 10).ToList();
            var weights = Enumerable.Repeat(0.1, 10).ToList();

            var result = calc.Calculate(points, weights);

            Assert.AreEqual(1, result);            
        }
    }
}
