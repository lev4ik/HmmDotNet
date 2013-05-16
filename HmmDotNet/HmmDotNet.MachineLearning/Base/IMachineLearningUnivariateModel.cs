namespace HmmDotNet.MachineLearning.Base
{
    public interface IMachineLearningUnivariateModel
    {
        /// <summary>
        ///     Likelihood of given model
        /// </summary>
        double Likelihood { get; }

        /// <summary>
        ///     Performs training procedure for the model
        /// </summary>
        /// <param name="observations"></param>
        void Train(double[] observations);
    }
}
