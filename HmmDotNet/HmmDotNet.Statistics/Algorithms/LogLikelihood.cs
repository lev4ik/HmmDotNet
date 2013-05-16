using System;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Distance;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.Statistics.Algorithms
{
    public static class LogLikelihood
    {
        /// <summary>
        ///     Caclulates Loglikelihood of Multivariate Gaussian Distribution
        /// </summary>
        /// <param name="observations">Observarion Sequence</param>
        /// <param name="covariance">Covariance Matrixe</param>
        /// <param name="mean">Mean Vector</param>
        /// <returns>Loglikelihood</returns>
        public static double Calculate(double[][] observations, double[,] covariance, double[] mean)
        {
            var N = observations.Length;
            var D = covariance.GetLength(0);

            var constant = N * D * 0.5 * Math.Log(2 * Math.PI);
            var covarianceMatrix = new Matrix(covariance);
            var covarianceMatrixInverse = covarianceMatrix.Inverse();

            var lnCovariance = N * 0.5 * Math.Log(covarianceMatrix.Determinant);
            var sumObservations = 0.0;

            for (int n = 0; n < N; n++)
            {
                var x = observations[n];
                var distance = Mahalanobis.Calculate(x, mean, covarianceMatrixInverse);
                sumObservations += distance;    
            }

            return - constant - lnCovariance - 0.5 * sumObservations;
        }

        /// <summary>
        ///     Calcualtes loglikelihood of Gaussian Mixture Distribution
        /// </summary>
        /// <param name="observations">Observation seuqnce</param>
        /// <param name="mixingCoefficients">Mixing coeficients</param>
        /// <param name="covariances">Array of covariance matrixes</param>
        /// <param name="mean">Array of mean vectors</param>
        /// <returns>Loglikelihood</returns>
        public static double Calculate(double[][] observations, double[] mixingCoefficients, double[][,] covariances, double[][] means)
        {
            var N = observations.Length;
            var K = mixingCoefficients.Length;
            var D = means[0].Length;

            var result = 0.0d;
            var constant = (1 / Math.Pow(Math.Sqrt(2 * Math.PI), D));

            for (int n = 0; n < N; n++)
            {
                var x = observations[n];
                var mixtureResult = 0.0;
                for (int k = 0; k < K; k++)
                {
                    var covarianceMatrix = new Matrix(covariances[k]);
                    var covarianceMatrixInverse = covarianceMatrix.Inverse();

                    var distance = Mahalanobis.Calculate(x, means[k], covarianceMatrixInverse);
                    var prefix = constant * (1 / Math.Sqrt(covarianceMatrix.Determinant));
                    mixtureResult += mixingCoefficients[k] * prefix * Math.Exp(-0.5 * distance);
                }

                result += Math.Log(mixtureResult);
            }

            return result;
        }

    }
}
