using System;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.MachineLearning.GeneralPredictors
{
    public class ValuePredictionErrorEstimator : IPredictionErrorEstimator<double>
    {
        #region Private Members

        private double[][] _actual;
        private double[][] _predicted;
        private double[][] _difference;
        private int N;
        private int D;

        #endregion Private Members

        public ValuePredictionErrorEstimator(double[][] actual, double[][] predicted)
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
            for (int n = 0; n < N; n++)
            {
                _difference[n] = new double[D];
                for (int d = 0; d < D; d++)
                {
                    _difference[n][d] = _actual[n][d] - _predicted[n][d];
                }
            }
        }

        #region Methods

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
                    mape[d] += Math.Abs(_difference[n][d] / _actual[n][d]);
                }
            }
            mape = mape.Product(100d / N);
            return mape.Round(2);
        }

        public double ReturnOnInvestment()
        {
            return 0d;
        }

        #endregion Methods
    }
}
