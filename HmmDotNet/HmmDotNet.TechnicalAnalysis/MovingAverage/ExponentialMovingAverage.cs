using System;
using System.Collections.Generic;
using System.Linq;

namespace HmmDotNet.TechnicalAnalysis.MovingAverage
{
    public class ExponentialMovingAverage
    {
        public double Calculate(IList<double> points, double alpha)
        {
            if (points == null)
            {
                throw new ArgumentException("Points can't be null");
            }

            if (alpha <= 0)
            {
                throw new ArgumentException("Alpha can't be negative");
            }

            var result = points.Select((point, i) => point * Math.Pow(alpha, (i + 1) - 1)).Sum();
            
            var denominator = 0.0;
            for (var i = 0; i < points.Count; i++)
            {
                denominator += Math.Pow(alpha, (i + 1) - 1);
            }

            return result / denominator;
        }
    }
}
