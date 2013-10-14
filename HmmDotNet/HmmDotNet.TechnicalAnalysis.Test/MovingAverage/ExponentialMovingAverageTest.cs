using System;
using System.Collections.Generic;
using System.Linq;
using HmmDotNet.TechnicalAnalysis.MovingAverage;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.TechnicalAnalysis.Test.MovingAverage
{
    [TestClass]
    public class ExponentialMovingAverageTest
    {
        [TestMethod]
        public void Calculate_PointsIsNull_ArgumentExceptionThrown()
        {
            var calc = new ExponentialMovingAverage();

            try
            {
                calc.Calculate(null, 0);
            }
            catch (Exception ex)
            {
                Assert.IsInstanceOfType(ex, typeof(ArgumentException));
            }
        }

        [TestMethod]
        public void Calculate_AlphaIsNegative_ArgumentExceptionThrown()
        {
            var calc = new ExponentialMovingAverage();
            var points = new List<double>();

            try
            {
                calc.Calculate(points, -1);
            }
            catch (Exception ex)
            {
                Assert.IsInstanceOfType(ex, typeof(ArgumentException));
            }
        }

        [TestMethod]
        public void Calculate_5PointsAllOneAndAlphaIsHalf_1()
        {
            var calc = new ExponentialMovingAverage();
            var points = Enumerable.Repeat(1.0, 5).ToList();
            var alpha = 0.5;

            var result = calc.Calculate(points, alpha);

            Assert.AreEqual(1d, result);
        }

        [TestMethod]
        public void Calculate_5PointsFrom1To5AndAlphaIsHalf_1()
        {
            var calc = new ExponentialMovingAverage();
            var points = new List<double>();
            var alpha = 0.5;
            for (int i = 0; i < 5; i++)
            {
                points.Add(i + 1);
            }

            var result = calc.Calculate(points, alpha);

            Assert.AreEqual(1.8387, Math.Round(result, 4));
        }
    }
}
