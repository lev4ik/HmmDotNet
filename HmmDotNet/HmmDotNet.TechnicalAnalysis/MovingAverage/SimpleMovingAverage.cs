using System;
using System.Collections.Generic;
using System.Linq;

namespace HmmDotNet.TechnicalAnalysis.MovingAverage
{
    public class SimpleMovingAverage
    {
        public double Calculate(IList<double> points)
        {
            if (points == null)
            {
                throw new ArgumentException("Points can't be null");
            }

            var result = points.Sum();

            return result / points.Count;
        }
    }
}
