using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.Mathematic;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors
{
    public class LikelihoodBasedPredictor : IHiddenMarkovModelPredictor
    {
        public IPredictionResult Predict<TDistribution>(IHiddenMarkovModelState<TDistribution> model, IPredictionRequest request)
            where TDistribution : IDistribution
        {
            var trainingSet = (double[][])request.TrainingSet.Clone();
            var predictions = new PredictionResult {Predicted = new double[request.NumberOfDays][]};

            for (int i = 0; i < request.NumberOfDays; i++)
            {
                var prediction = PredictNextValue(model, request, trainingSet);
                if (request.NumberOfDays > 1)
                {
                    trainingSet = new double[request.TrainingSet.Length + i + 1][];
                    request.TrainingSet.CopyTo(trainingSet, 0);
                    for (var j = 0; j < i + 1; j++)
                    {
                        trainingSet[request.TrainingSet.Length + j] = request.TestSet[j];
                    }
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

        #region Private Methods

        private IList<ObservationWithLikelihood<double[]>> FindMostSimilarObservations<TDistribution>(IHiddenMarkovModelState<TDistribution> model, double[][] trainingSet, double yesterdayLikelihood, double tolerance)
            where TDistribution : IDistribution
        {
            var N = trainingSet.Length;
            var guessess = new List<ObservationWithLikelihood<double[]>>();
            var forwardBackward = new ForwardBackward(model.Normalized);

            for (var n = N - 2; n > 0; n--)
            {
                var x = Helper.Convert(new[] { trainingSet[n] });
                var likelihood = forwardBackward.RunForward(x, model);
                //Debug.Write((new Vector(observations[n])).ToString() + " : " + likelihood + " " + Environment.NewLine);

                if (Math.Abs(yesterdayLikelihood) - tolerance < Math.Abs(likelihood) && Math.Abs(yesterdayLikelihood) + tolerance > Math.Abs(likelihood))
                {
                    guessess.Add(new ObservationWithLikelihood<double[]>() { LogLikelihood = likelihood, Observation = trainingSet[n], PlaceInSequence = n - 1 });
                }
            }

            return guessess;
        }

        private ObservationWithLikelihood<double[]> FindBestGuess(IPredictionRequest request, IList<ObservationWithLikelihood<double[]>> guessess)
        {
            var result = new ObservationWithLikelihood<double[]>() { Observation = new double[request.TrainingSet[0].Length] };
            var maxLogLikelihood = double.NegativeInfinity;

            for (var i = 0; i < guessess.Count; i++)
            {
                if (maxLogLikelihood < guessess[i].LogLikelihood)
                {
                    maxLogLikelihood = guessess[i].LogLikelihood;
                    result.LogLikelihood = guessess[i].LogLikelihood;
                    result.Observation = guessess[i].Observation;
                    result.PlaceInSequence = guessess[i].PlaceInSequence;
                    result.NumberOfGuesses = guessess.Count;
                }
            }

            Debug.Write(result.PlaceInSequence + " : " + (new Vector(result.Observation)) + " " + result.NumberOfGuesses + " " + Environment.NewLine);

            return result;
        }

        private double[] PredictNextValue<TDistribution>(IHiddenMarkovModelState<TDistribution> model, IPredictionRequest request, double[][] trainingSet)
            where TDistribution : IDistribution
        {
            var N = trainingSet.Length;
            var K = trainingSet[0].Length;
            var result = new double[K];

            var yesterday = trainingSet[N - 1];
            var forwardBackward = new ForwardBackward(model.Normalized);
            var yesterdayLikelihood = forwardBackward.RunForward(Helper.Convert(new[] { yesterday }), model);

            Debug.WriteLine("Yesterday Likelihood : " + new Vector(yesterday) + " : " + yesterdayLikelihood + " ");

            var guessess = FindMostSimilarObservations(model, trainingSet, yesterdayLikelihood, request.Tolerance);
            var bestGuessPlace = FindBestGuess(request, guessess);
            var tomorrow = trainingSet[bestGuessPlace.PlaceInSequence + 1];
            var mostSimilar = trainingSet[bestGuessPlace.PlaceInSequence];

            for (var k = 0; k < K; k++)
            {
                if (bestGuessPlace.PlaceInSequence != trainingSet.Length)
                {
                    result[k] = yesterday[k] + (tomorrow[k] - mostSimilar[k]);
                }
            }

            Debug.WriteLine("Predicted (for day " + N +") : " + new Vector(result) + " : " + forwardBackward.RunForward(Helper.Convert(result), model));

            return result;
        }


        #endregion Private Methods
    }
}
