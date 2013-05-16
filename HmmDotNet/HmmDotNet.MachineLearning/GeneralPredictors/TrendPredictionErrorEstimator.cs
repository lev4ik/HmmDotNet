using System;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.MachineLearning.GeneralPredictors
{
    public enum TrendDirection
    {
        Down,
        NoChange,
        Up
    }

    public class TrendPredictionErrorEstimator : IPredictionErrorEstimator<double>
    {
        #region Private Members

        private double[][] _actual;
        private double[][] _predicted;
        private double[][] _difference;
        private int N;
        private int D;

        #endregion Private Members

        public TrendPredictionErrorEstimator(double[][] actual, double[][] predicted)
        {
            _actual = actual;
            _predicted = predicted;
            if (_actual.Length != _predicted.Length)
            {
                throw new ApplicationException("Actual and Predicted series length must be the same");
            }
            N = _actual.Length;
            D = _actual[0].Length;

            _difference = new double[N][];
            // Initialize with no difference in trend
            _difference[0] = new double[D];
            for (int d = 0; d < D; d++)
            {
                _difference[0][d] = 0d;
            }
            for (int n = 1; n < N; n++)
            {
                _difference[n] = new double[D];
                for (int d = 0; d < D; d++)
                {
                    var a = GetTrendDirectionFromObservations(_actual[n - 1][d], _actual[n][d]);
                    var p = GetTrendDirectionFromObservations(_predicted[n - 1][d], _predicted[n][d]);
                    // if actual and previous are equal than trend predicted correctly
                    _difference[n][d] = (a == p) ? 0d : 1d;
                }
            }
        }

        #region Private Methods

        private TrendDirection GetTrendDirectionFromObservations(double previous, double current)
        {
            if (current > previous)
            {
                return TrendDirection.Up;
            } 
            if (current < previous)
            {
                return TrendDirection.Down;
            }
            return TrendDirection.NoChange;
        }

        #endregion Private Methods

        public double[] CumulativeForecastError()
        {
            var cfe = new double[D];
            for (int n = 0; n < N; n++)
            {
                for (int d = 0; d < D; d++)
                {
                    cfe[d] += _difference[n][d];
                }
            }
            return cfe.Round(2);
        }

        public double[] MeanError()
        {
            var me = CumulativeForecastError();
            me = me.Product(1d / N);
            return me.Round(2);
        }

        public double[] MeanSquaredError()
        {
            var mfe = new double[D];
            for (int n = 0; n < N; n++)
            {
                for (int d = 0; d < D; d++)
                {
                    mfe[d] += Math.Pow(_difference[n][d], 2);
                }
            }
            mfe = mfe.Product(1d / N);
            return mfe.Round(2);
        }

        public double[] RootMeanSquaredError()
        {
            var mfe = MeanSquaredError();
            var rmfe = new double[D];
            for (int d = 0; d < D; d++)
            {
                rmfe[d] += Math.Sqrt(mfe[d]);
            }
            return rmfe.Round(2);
        }

        public double[] MeanAbsoluteDeviation()
        {
            var mad = new double[D];
            for (int n = 0; n < N; n++)
            {
                for (int d = 0; d < D; d++)
                {
                    mad[d] += Math.Abs(_difference[n][d]);
                }
            }
            mad = mad.Product(1d / N);
            return mad.Round(2);
        }

        public double[] MeanAbsolutePercentError()
        {
            var mape = new double[D];
            for (int n = 0; n < N; n++)
            {
                for (int d = 0; d < D; d++)
                {
                    mape[d] += _difference[n][d];
                }
            }
            double denominator = 100d/N;
            mape = mape.Product(denominator);
            return mape.Round(2);
        }

        public double ReturnOnInvestment()
        {
            var result = 0d;
            var lastBuyPrice = _actual[0][D - 1];
            for (var t = 1; t < _actual.Length; t++)
            {
                var p = GetTrendDirectionFromObservations(_predicted[t - 1][D - 1], _predicted[t][D - 1]);
                if (p == TrendDirection.Down)
                {
                    // sell
                    if (!lastBuyPrice.EqualsToZero())
                    {
                        result += _actual[t][D - 1] - lastBuyPrice;
                        lastBuyPrice = 0d;
                    }
                }
                else
                {
                    // buy
                    if (lastBuyPrice.EqualsToZero())
                    {
                        lastBuyPrice = _actual[t][D - 1];
                    }
                }
            }
            return Math.Round(result, 2);
        }
    }
}
