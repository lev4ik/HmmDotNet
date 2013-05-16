// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ForwardBackward.cs" company="">
//   
// </copyright>
// <summary>
//   Base class for Forward-Backward procedure
// </summary>
// --------------------------------------------------------------------------------------------------------------------
namespace HmmDotNet.MachineLearning.Algorithms
{
    using System;
    
    /// <summary>
    ///     Base class for Forward-Backward procedure
    /// </summary>
    public abstract class BaseForwardBackward : IDynamicProgrammingAlgorithm
    {
        protected bool LogNormalized;

        #region IDynamicProgrammingAlgorithm Members

        /// <summary>
        ///     Gets or sets Computation path history
        /// </summary>
        public double[][] ComputationPath { get; protected set; }

        /// <summary>
        ///     Prints Computational Path
        /// </summary>
        public void PrintComputationalPath()
        {
            if (ComputationPath != null)
            {
                var T = ComputationPath.Length;
                var N = ComputationPath[0].Length;

                for (var t = 0; t < T; t++)
                {
                    for (var i = 0; i < N; i++)
                    {
                        Console.Write(string.Format("[{0}|{1:0.00000000000}]", i, ComputationPath[t][i]));
                    }

                    Console.WriteLine();
                }
            }
        }

        #endregion
    }
}
