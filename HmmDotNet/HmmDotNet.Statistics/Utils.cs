using System;

namespace HmmDotNet.Statistics
{
    public static class Utils
    {
        #region Mean Calculations

        public static double Mean(double[] observations)
        {
            var sum = 0.0;
            for (var i = 0; i < observations.Length; i++)
            {
                sum = sum + observations[i];
            }
            return sum / observations.Length;
        }

        public static double Mean(double[] observations, double[] weights)
        {
            var sum = 0.0;
            for (var i = 0; i < observations.Length; i++)
            {
                sum = sum + observations[i] * weights[i];
            }
            return sum;
        }

        /// <summary>
        ///     Mean of k-dimentional n vectors. To calculate simple mean 
        ///     pass weights vector with 1/k in each cell. When each dimesion is a column in
        ///     the matrix the mean is calculated by summing column i value over each row.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static double[] Mean(double[][] observations, double[] weights)
        {
            var rows = observations.Length;
            var cols = observations[0].Length;
            var mean = new double[cols];
            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    mean[i] += observations[j][i] * weights[j];
                }
            }
            return mean;
        }

        #endregion Mean Calculations

        #region Covariance Calculation

        /// <summary>
        ///     Calculates covariance matrix
        /// <para>
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://en.wikipedia.org/wiki/Covariance_matrix">Covariance Matrix</a></description></item>
        ///   </list>
        /// </para>
        /// </summary>
        /// <param name="observations">Observation vectors</param>
        /// <param name="means">Mean vector of all variables in observation vectors</param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static double[,] Covariance(double[][] observations, double[] means, double[] weights)
        {
            var rows = observations.Length;
            var cols = observations[0].Length;
            var cov = new double[means.Length, means.Length];

            for (var colsx = 0; colsx < cols; colsx++)
            {
                for (var colsy = colsx; colsy < cols; colsy++)
                {
                    var x = new double[rows];
                    var y = new double[rows];
                    for (var j = 0; j < rows; j++)
                    {
                        x[j] = observations[j][colsx];
                        y[j] = observations[j][colsy];
                    }
                    cov[colsx, colsy] = cov[colsy, colsx] = Covariance(x, y, means[colsx], means[colsy]);
                }                
            }
            return cov;
        }

        public static double[,] Covariance(double[][] observations)
        {
            var weights = new double[observations[0].Length];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 1.0d / weights.Length;
            }
            return Covariance(observations, Mean(observations, weights), weights);
        }

        /// <summary>
        ///     Calculates Covariance between two variables. If the expectation is estimated than devide by n-1 because the 
        ///     actual observation is in sequence, otherwise devide by n.
        /// <para>
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://en.wikipedia.org/wiki/Covariance">Covariance</a></description></item>
        ///   </list>
        /// </para>
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="meanX"></param>
        /// <param name="meanY"></param>
        /// <returns></returns>
        public static double Covariance(double[] x, double[] y, double meanX, double meanY)
        {
            if (x.Length != y.Length)
            {
                throw new ApplicationException("Observation vectors must be of the same length");
            }
            var sum = 0.0;
            var n = x.Length;
            for (var i = 0; i < n; i++)
            {
                sum = sum + (x[i] - meanX) * (y[i] - meanY);
            }
            return sum / n;// (n - 1);
        }

        #endregion Covariance Calculation

        #region Variance Calculations

        public static double Variance(double[] observations, double mean)
        {
            var sum = 0.0;
            for (var i = 0; i < observations.Length; i++)
            {
                sum = sum + Math.Pow(observations[i] - mean, 2);
            }
            return sum / observations.Length;
        }

        public static double Variance(double[] observations, double[] weights, double mean)
        {
            var sum = 0.0;
            for (var i = 0; i < observations.Length; i++)
            {
                sum = sum + Math.Pow( (observations[i] - mean) * weights[i], 2);
            }
            return sum;
        }

        #endregion Variance Calculations
    }
}
