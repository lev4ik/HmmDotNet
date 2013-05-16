namespace HmmDotNet.Mathematic.MatrixDecomposition
{
    /// <summary>
    ///     Define generic interface with type parameter that must be value type.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IDecomposition<T> where T : struct
    {
        #region Properties

        /// <summary>
        ///     Matrix decomposition calculated for farther use
        /// </summary>
        T[,] Decomposition { get; }
        T[,] Upper { get; }
        T[,] Lower { get; }

        #endregion Properties

        #region Methods

        /// <summary>
        ///     Calculates the decomposition
        /// </summary>
        /// <param name="m">Matrix on wich decomposition will be calcuated</param>
        void Calculate(T[,] m);
        /// <summary>
        ///     Calculates determinant based on precalculated decomposition
        /// </summary>
        /// <returns></returns>
        double Determinant();
        /// <summary>
        ///     Calculates inverted determinant based on precalculated decomposition
        /// </summary>
        /// <returns></returns>
        double InvertDeterminant();
        /// <summary>
        ///     Calculates inverse matrix using the decomposition results
        /// </summary>
        /// <returns></returns>
        /// <param name="m">Matrix on wich decomposition will be calcuated</param>
        T[,] Inverse(T[,] m);

        #endregion Methods
    }
}
