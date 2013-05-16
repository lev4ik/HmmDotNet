namespace HmmDotNet.Mathematic
{
    public interface IVector<T> where T : struct
    {
        #region Properties

        /// <summary>
        ///     Number of dimentions in the vector
        /// </summary>
        int Dimention { get; }
        /// <summary>
        ///     Instance of the matrix
        /// </summary>
        T[] V { get; }

        #endregion Properties

        #region Methods

        /// <summary>
        ///     Calculates product of two vectors. Product 1xN * Nx1 = a
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        T Product(T[] v);
        /// <summary>
        ///     Calculates product of vector and scalar. Product 1xN * a = 1xN
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        T[] Product(T x);
        /// <summary>
        ///     Add's two vectors
        /// </summary>
        /// <param name="v">vector</param>
        /// <returns></returns>
        T[] Add(T[] v);
        /// <summary>
        ///     Add's vector with scalar
        /// </summary>
        /// <param name="x">scalar</param>
        /// <returns></returns>
        T[] Add(T x);
        /// <summary>
        ///     Substructs one vector from another
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        T[] Substract(T[] v);
        /// <summary>
        ///     Substructs scalar from vector
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        T[] Substract(T x);
        /// <summary>
        ///     Caclculates outer product of two vectors. Outer Product Nx1 * 1xN = NxN
        /// </summary>
        /// <param name="v">vector</param>
        /// <returns></returns>
        T[,] OuterProduct(T[] v);

        #endregion Methods
    }
}
