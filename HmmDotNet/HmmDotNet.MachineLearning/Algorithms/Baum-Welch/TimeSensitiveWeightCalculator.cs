using System;

namespace HmmDotNet.MachineLearning.Algorithms.Baum_Welch
{
    public static class TimeSensitiveWeightCalculator
    {
        public static double[] Calculate(int k, int T)
        {
            var result = new double[T];
            var r = 2d / (k + 1);
            for (var t = 0; t < T - 1; t++)
            {
                if (t <= k - 1)
                {
                    result[t] = Math.Pow(1 - r, T - k) * (1d / k);
                }
                else
                {
                    result[t] = r * Math.Pow(1 - r, T - t);
                }
            }
            result[T - 1] = r;
            return result;
        }
    }
}
