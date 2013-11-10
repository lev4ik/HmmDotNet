using System;
using System.Collections.Generic;
using System.Linq;

namespace HmmDotNet.TechnicalAnalysis.MovingAverage
{
    public class ExponentialMovingAverage
    {
        public double Calculate(double yesterdayMovingAverage, double todayValue, int numberOfDays)
        {
            var alpha = 2d / (numberOfDays + 1);
            return yesterdayMovingAverage * (1 - alpha) + alpha * todayValue;
        }

        public IList<double> Calculate(IList<double> points, int numberOfDays)
        {
            if (points == null)
            {
                throw new ArgumentException("Points can't be null");
            }

            if (points.Count < numberOfDays)
            {
                throw new ArgumentException("Points should have more length than number of days");
            }

            var result = new List<double>();
            var sma = new SimpleMovingAverage();
            var yesterdayMovingAverage = sma.Calculate(points.Take(numberOfDays).ToList());
            for (var i = numberOfDays; i < points.Count; i++)
            {
                var ema = Calculate(yesterdayMovingAverage, points[i], numberOfDays);
                result.Add(ema);
                yesterdayMovingAverage = ema;
            }
            return result;
        }
    }
}
