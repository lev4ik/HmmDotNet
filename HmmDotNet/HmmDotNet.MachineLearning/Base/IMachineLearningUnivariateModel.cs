namespace HmmDotNet.MachineLearning.Base
{
    public interface IMachineLearningUnivariateModel
    {
        /// <summary>
        ///     Performs training procedure for the model
        /// </summary>
        /// <param name="observations"></param>
        void Train(double[] observations);
    }
}
