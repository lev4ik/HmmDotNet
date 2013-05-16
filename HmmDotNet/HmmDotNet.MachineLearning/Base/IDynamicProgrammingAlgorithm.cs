namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     Definition of most general dynamic programming algorithm behavior
    /// </summary>
    /// <typeparam name="K">Index value of the resulting data staructure</typeparam>
    /// <typeparam name="T">Result set per each index</typeparam>
    public interface IDynamicProgrammingAlgorithm
    {
        /// <summary>
        ///     Lattice representing calculation's path
        /// </summary>
        double[][] ComputationPath { get; }

        /// <summary>
        ///     Prints all the computation results' in the calculation path structure
        /// </summary>
        void PrintComputationalPath();
    }
}
