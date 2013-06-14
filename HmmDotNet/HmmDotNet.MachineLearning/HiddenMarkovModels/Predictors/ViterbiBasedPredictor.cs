using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors
{
    public class ViterbiBasedPredictor : IHiddenMarkovModelPredictor
    {
        public int NumberOfIterations { get; set; }
        public double LikelihoodTolerance { get; set; }

        /// <summary>
        ///     Viterbi based prediction algorithm. For each observation in test set we perform following action :
        ///     1. Run Viterbi and find best state that generated last day observation
        ///     2. Prediction for next day is weighted average of means of predicting state emissions
        ///     3. Re-Train the model with new value added from training set
        ///     4. Return to step 1
        /// </summary>
        /// <typeparam name="TDistribution"></typeparam>
        /// <param name="model"></param>
        /// <param name="request"></param>
        /// <returns></returns>
        public IPredictionResult Predict<TDistribution>(IHiddenMarkovModel<TDistribution> model, IPredictionRequest request) where TDistribution : IDistribution
        {
            NumberOfIterations = request.NumberOfTrainingIterations;
            LikelihoodTolerance = request.TrainingLikelihoodTolerance;
            var predictions = new PredictionResult { Predicted = new double[request.NumberOfDays][] };
            var trainingSet = (double[][])request.TrainingSet.Clone();
            
            for (int i = 0; i < request.NumberOfDays; i++)
            {
                var prediction = PredictNextValue(model, trainingSet);
                if (request.NumberOfDays > 1)
                {
                    trainingSet = new double[request.TrainingSet.Length + i + 1][];
                    request.TrainingSet.CopyTo(trainingSet, 0);
                    for (var j = 0; j < i + 1; j++)
                    {
                        trainingSet[request.TrainingSet.Length + j] = request.TestSet[j];
                    }
                    var iterations = NumberOfIterations;
                    ((IMachineLearningMultivariateModel)model).Train(trainingSet, iterations, LikelihoodTolerance);
                }
                predictions.Predicted[i] = prediction;
            }

            return predictions;
        }

        public IEvaluationResult Evaluate(IEvaluationRequest request)
        {
            var errorEstimator = PredictionErrorEstimatorFactory.GetErrorEstimator(request.EstimatorType,
                                                                                   request.PredictionParameters.TestSet,
                                                                                   request.PredictionToEvaluate
                                                                                          .Predicted);
            var result = new PredictionResultEvaluation();
            result.CumulativeForecastError = errorEstimator.CumulativeForecastError();
            result.MeanAbsoluteDeviation = errorEstimator.MeanAbsoluteDeviation();
            result.MeanAbsolutePercentError = errorEstimator.MeanAbsolutePercentError();
            result.MeanError = errorEstimator.MeanError();
            result.MeanSquaredError = errorEstimator.MeanSquaredError();
            result.ReturnOnInvestment = errorEstimator.ReturnOnInvestment();
            result.RootMeanSquaredError = errorEstimator.RootMeanSquaredError();

            return result;
        }

        #region Private Method

        private double[] PredictNextValue<TDistribution>(IHiddenMarkovModel<TDistribution> model, double[][] trainingSet) where TDistribution : IDistribution
        {
            var alg = new Viterbi(model.Normalized);
            var mpp = alg.Run(Helper.Convert(trainingSet), model.GetStates(), model.Pi, model.TransitionProbabilityMatrix, model.GetEmissions());

            var emission = model.Emission[mpp[trainingSet.Length - 1].Index];
            var prediction = CalculatePredictionValue(emission, trainingSet);

            return prediction;
        }

        private double[] CalculatePredictionValue<TDistribution>(TDistribution emission, double[][] trainingSet) where TDistribution : IDistribution
        {
            var result = new double[trainingSet[0].Length];
            if (typeof(TDistribution).Name == "Mixture`1")
            {
                switch (typeof(TDistribution).GetGenericArguments()[0].FullName)
                {
                    case "TA.Statistics.Distributions.IMultivariateDistribution":
                        var e = emission as Mixture<IMultivariateDistribution>;
                        result = e.Mean;
                        break;
                }
            }
            else
            {
                switch (typeof(TDistribution).FullName)
                {
                    case "TA.Statistics.Distributions.IMultivariateDistribution":
                        result = ((IMultivariateDistribution)emission).Mean;
                        break;
                }
            }
            return result;
        }

        #endregion Private Method
    }
}
