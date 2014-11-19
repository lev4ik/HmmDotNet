using System;

namespace HmmDotNet.MachineLearning.Algorithms.Baum_Welch
{
    public static class TimeSensitiveWeightCalculator
    {
        public static decimal[] Calculate(int k, int T)
        {
            var result = new decimal[T];
            var r = 2m / (k + 1);
            for (var t = 0; t < T - 1; t++)
            {
                if (t <= k - 1)
                {
                    result[t] = (decimal)Math.Pow(1 -(double)r, T - k) * (1m / k);
                }
                else
                {
                    result[t] = r * (decimal)Math.Pow(1 - (double)r, T - t);
                }
            }
            result[T - 1] = r;
            return result;
        }
    }
}
