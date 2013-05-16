using System;

namespace HmmDotNet.Mathematic.MatrixDecomposition
{
    public class Cholesky : IDecomposition<double>
    {
        private double[,] _decomposition;
        private double[,] _upper = null;
        private double[,] _lower = null;

        public double[,] Decomposition
        {
            get { return _decomposition; }
        }

        public double[,] Upper
        {
            get { return _upper; }
        }

        public double[,] Lower
        {
            get { return _lower; }
        }

        /// <summary>
        ///     Returns matrix decomposition of m, when the decomposition is formed form 
        ///     matrix U and its transpose part. To calculate this decomposition matrix
        ///     m must be positive definite
        /// <para>    
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://fedc.wiwi.hu-berlin.de/xplore/ebooks/html/csa/node36.html">Cholesky Decomposition</a></description></item>
        ///   </list></para>
        /// </remarks>
        /// </summary>
        /// <param name="m"></param>
        public void Calculate(double[,] m)
        {
            var n = m.GetLength(0);
            var matrix = new Matrix(m);
            _decomposition = matrix.Identity();

            for (int i = 0; i < n; i++)
            {
                var sum = 0.0;
                for (int k = 0; k < i; k++)
                {
                    sum += _decomposition[k, i] * _decomposition[k, i];
                }
                _decomposition[i, i] = Math.Sqrt(m[i, i] - sum);

                for (int j = i + 1; j < n; j++)
                {
                    sum = 0.0;
                    for (int k = 0; k < i; k++)
                    {
                        sum += _decomposition[k, i] * _decomposition[k, j];
                    }
                    _decomposition[i, j] = (m[i, j] - sum) / _decomposition[i, i];
                    _decomposition[j, i] = _decomposition[i, j];
                }
            }           
        }

        /// <summary>
        ///     Calculates determinant of the matrix using precalculated decomposition.
        /// <para>
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://www.proofwiki.org/wiki/Determinant_of_a_Triangular_Matrix">Determinant of a Triangular Matrix</a></description></item>
        ///   </list></para>
        /// </remarks>
        /// </summary>
        /// <returns></returns>
        public double Determinant()
        {
            if (_decomposition == null)
            {
                throw new ApplicationException("Decomposition matrix not calculated");
            }
            var n = _decomposition.GetLength(0);
            var product = 1.0;
            for (int i = 0; i < n; i++)
            {
                product = product * _decomposition[i, i];
            }
            return product;
        }


        public double InvertDeterminant()
        {
            throw new NotImplementedException();
        }

        public double[,] Inverse(double[,] m)
        {
            throw new NotImplementedException();
        }
    }
}
