using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors
{
    public interface IHiddenMarkovModelPredictor
    {
        /// <summary>
        ///     Given trained model and prediction request this method runs prediction algroithms
        ///     and returnes prediction result
        /// </summary>
        /// <typeparam name="M">Type of trained model for prediction</typeparam>
        /// <typeparam name="T">Type of observation</typeparam>
        /// <param name="model">Trained model for prediction</param>
        /// <param name="request">Prediction request data structure</param>
        /// <returns></returns>
        IPredictionResult Predict<TDistribution>(IHiddenMarkovModel<TDistribution> model, IPredictionRequest request)
            where TDistribution : IDistribution;

        /// <summary>
        ///     Evaluated prediction result and returns evaluation result data structure5
        /// </summary>
        /// <typeparam name="T">Type of observation</typeparam>
        /// <param name="request">Evaluation request data structure</param>
        /// <returns></returns>
        IEvaluationResult Evaluate(IEvaluationRequest request);
    }
}
