namespace HmmDotNet.Mathematic
{
    public interface IMatrix<T> where T : struct
    {
        #region Properties

        /// <summary>
        ///     Number of rows in the matrix
        /// </summary>
        int Rows { get; }
        /// <summary>
        ///     Number of columns in the matrix
        /// </summary>
        int Columns { get; }
        /// <summary>
        ///     Return determinant of the matrix
        /// </summary>
        double Determinant { get; }
        /// <summary>
        ///     Returns determinant of the inverse matrix
        /// </summary>
        double DeterminantInverse { get; }
        /// <summary>
        ///     Instance of the matrix
        /// </summary>
        T[,] M { get; }

        #endregion Properties

        #region Methods

        /// <summary>
        ///     Creates identity matrix of given size
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        T[,] Identity();
        /// <summary>
        ///     Calculates product of two matrixes
        /// </summary>
        /// <param name="B"></param>
        /// <returns></returns>
        T[,] Product(T[,] B);
        /// <summary>
        ///     Calculates product of vector with matrix
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        T[] Product(T[] b);
        /// <summary>
        ///     Calculates product of matrix and scalar
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        T[,] Product(T x);
        /// <summary>
        ///     Calculates transpose of given matrix
        /// </summary>
        /// <returns></returns>
        T[,] Transpose();
        /// <summary>
        ///     Calculates inverse of given matrix
        /// </summary>
        /// <returns></returns>
        T[,] Inverse();
        /// <summary>
        ///     Calculates diagonal of given square matrix
        /// </summary>
        /// <returns></returns>
        T[] Diagonal();

        #endregion Methods
    }
}
