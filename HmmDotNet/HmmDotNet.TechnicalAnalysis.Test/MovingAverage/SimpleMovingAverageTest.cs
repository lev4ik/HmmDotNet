using System;
using System.Collections.Generic;
using System.Linq;
using HmmDotNet.TechnicalAnalysis.MovingAverage;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.TechnicalAnalysis.Test.MovingAverage
{
    [TestClass]
    public class SimpleMovingAverageTest
    {
        [TestMethod]
        public void Calculate_PointsIsNull_ArgumentExceptionThrown()
        {
            var calc = new SimpleMovingAverage();

            try
            {
                calc.Calculate(null);
            }
            catch (Exception ex)
            {
                Assert.IsInstanceOfType(ex, typeof(ArgumentException));
            }                        
        }

        [TestMethod]
        public void Calculate_10PointsAllOnes_1()
        {
            var calc = new SimpleMovingAverage();
            var points = Enumerable.Repeat(1d, 10).ToList();

            var result = calc.Calculate(points);

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void Calculate_10PointsFromOneToTen_FiveAndAHalf()
        {
            var calc = new SimpleMovingAverage();
            var points = new List<double>();
            for (int i = 0; i < 10; i++)
            {
                points.Add(i + 1);
            }

            var result = calc.Calculate(points);

            Assert.AreEqual(5.5, result);
        }
    }
}
