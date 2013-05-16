using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Mathematic.Distance
{
    /// <summary>
    ///     Mahalanobis
    /// </summary>
    public class Mahalanobis
    {
        /// <summary>
        ///     Calculates Mahalanobis distance
        /// </summary>
        /// <param name="x">Observation vector</param>
        /// <param name="m">Mean vector</param>
        /// <param name="sigmaInverse">Covariance matrix</param>
        /// <returns>Disntance</returns>
        public static double Calculate(double[] x, double[] m, double[,] sigmaInverse)
        {
            var distance = 0.0d;
            var rows = x.Length;
            var columns = x.Length;
            var temp = new double[rows];
            var z = x.Substruct(m);
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++)
                {
                    temp[i] += sigmaInverse[i, j] * z[j];
                }
            }

            distance = z.Product(temp);
            return distance;
        }
    }
}
