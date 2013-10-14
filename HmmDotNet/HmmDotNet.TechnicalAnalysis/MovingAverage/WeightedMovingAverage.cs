using System;
using System.Collections.Generic;
using System.Linq;

namespace HmmDotNet.TechnicalAnalysis.MovingAverage
{
    public class WeightedMovingAverage
    {
        public double Calculate(IList<double> points, IList<double> weights)
        {
            if (points == null)
            {
                throw new ArgumentException("Points can't be null");
            }

            if (weights == null)
            {
                throw new ArgumentException("Weights can't be null");
            }

            if (points.Count != weights.Count)
            {
                throw new ArgumentException("Points not equals to null");
            }

            var result = points.Select((point, i) => point * weights[i]).Sum();

            return result / weights.Sum();
        }
    }
}
