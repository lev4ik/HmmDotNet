namespace HmmDotNet.MachineLearning.Base
{
    public interface IMachineLearningMultivariateModel
    {
        /// <summary>
        ///     Likelihood of given model
        /// </summary>
        double Likelihood { get; }

        /// <summary>
        ///     Performs training procedure for the model
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="numberOfIterations"></param>
        /// <param name="likelihoodTolerance"></param>
        void Train(double[][] observations, int numberOfIterations, double likelihoodTolerance);
    }
}
