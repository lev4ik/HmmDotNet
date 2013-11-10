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
        public void Calculate_PointsHasLessThanNumberOfDays_ArgumentExceptionThrown()
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
        public void Calculate_YesterdayEma10AndTodayPrice10And9Days_10()
        {
            var calc = new ExponentialMovingAverage();
            var yesterdayEma = 10;
            var today = 10;
            var numberOfDays = 9;

            var result = calc.Calculate(yesterdayEma, today, numberOfDays);

            Assert.AreEqual(10, result);
        }

        [TestMethod]
        public void Calculate_20PointsAnd10Days_11DaysEMA()
        {
            var points = new List<double>()
                {
                    22.2734,
                    22.194,
                    22.0847,
                    22.1741,
                    22.184,
                    22.1344,
                    22.2337,
                    22.4323,
                    22.2436,
                    22.2933,
                    22.1542,
                    22.3926,
                    22.3816,
                    22.6109,
                    23.3558,
                    24.0519,
                    23.753,
                    23.8324,
                    23.9516,
                    23.6338
                };
            var calc = new ExponentialMovingAverage();

            var ema = calc.Calculate(points, 10);

            Assert.AreEqual(22.2119, Math.Round(ema[0], 4));
            Assert.AreEqual(22.2448, Math.Round(ema[1], 4));
            Assert.AreEqual(22.2697, Math.Round(ema[2], 4));
            Assert.AreEqual(22.3317, Math.Round(ema[3], 4));
            Assert.AreEqual(22.5179, Math.Round(ema[4], 4));
            Assert.AreEqual(22.7968, Math.Round(ema[5], 4));
            Assert.AreEqual(22.9707, Math.Round(ema[6], 4));
            Assert.AreEqual(23.1273, Math.Round(ema[7], 4));
            Assert.AreEqual(23.2772, Math.Round(ema[8], 4));
            Assert.AreEqual(23.3420, Math.Round(ema[9], 4));            
        }
    }
}
