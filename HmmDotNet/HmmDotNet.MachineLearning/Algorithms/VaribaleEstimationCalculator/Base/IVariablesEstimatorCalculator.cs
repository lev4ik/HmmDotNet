namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     Calculates variables for parameters estimation
    /// </summary>
    /// <typeparam name="T">Distribution</typeparam>
    /// <typeparam name="K">Observation</typeparam>
    public interface IVariablesEstimatorCalculator<T, K>
    {
        /// <summary>
        ///     Number of states
        /// </summary>
        int N { get; set; }



        /// <summary>
        ///     Probability of being in state i at time t
        /// </summary>
        /// <param name="i">State</param>
        /// <param name="t">Time</param>
        /// <param name="alpha">Forward variable</param>
        /// <param name="beta">Backward variable</param>
        /// <returns></returns>
        double Gamma(int i, int t, double[][] alpha, double[][] beta);
        /// <summary>
        ///     Probability of being in state i at time t and in state j in time t+1
        /// </summary>
        /// <param name="i">Start state</param>
        /// <param name="j">End state</param>
        /// <param name="t">Time</param>
        /// <param name="transitionProbability">Transition Probability</param>       
        /// <param name="x">Observation in time t</param>
        /// <param name="alpha">Forward variable</param>
        /// <param name="beta">Backward variable</param>
        /// <param name="b">Distribution function in state i</param>
        /// <returns></returns>
        double Ksi(int i, int j, int t, double transitionProbability, K x, double[][] alpha, double[][] beta, T b);
        /// <summary>
        ///     Probability that the l-th component of the i-th mixture generated observation o(t)
        /// </summary>
        /// <param name="l">Component</param>
        /// <param name="x">Observation in time t</param>
        /// <param name="gamma">Probability of being in state i at time t</param>
        /// <param name="c">Weight of l-th component in state i</param>
        /// <param name="b">Distribution function</param>
        /// <returns></returns>
        double ComponentGamma(int l, K x, double gamma, double c, T b);
    }
}
