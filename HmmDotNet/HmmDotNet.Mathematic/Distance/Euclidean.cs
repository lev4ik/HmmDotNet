using System;

namespace HmmDotNet.Mathematic.Distance
{
    /// <summary>
    ///     Euclidean
    /// </summary>
    public class Euclidean
    {
        /// <summary>
        ///     Calculates Euclidean distance
        /// </summary>
        /// <param name="x">Vector</param>
        /// <param name="y">Vector</param>
        /// <returns>Distance</returns>
        public static double Calculate(double[] x, double[] y)
        {
            var distance = 0.0d;
            var n = x.Length;
            var sum = 0.0d;
            for (var i = 0; i < n; i++)
            {
                sum += Math.Pow(x[i] - y[i], 2);
            }

            distance = Math.Sqrt(sum);
            return distance;
        }
    }
}
