using System;
using HmmDotNet.Extentions.Base;
using HmmDotNet.Extentions.Data;

namespace HmmDotNet.Extentions
{
    public class ChangeRatioFinder : IChangeRatioFinder
    {
        public MaximumChangeRatios GetMaximumChangeRatios(double[][] array)
        {
            var result = new MaximumChangeRatios { Down = double.MinValue, Up = double.MinValue };
            for (var i = 0; i < array.Length - 1; i++)
            {
                var diff = CalculatePercentChange(array[i][3], array[i + 1][3]);
                var checksum = array[i + 1][3] - array[i][3];
                //  down
                if (checksum < 0)
                {
                    if (result.Down < Math.Abs(diff))
                    {
                        result.Down = Math.Abs(diff);
                    }
                }
                //  up
                else
                {
                    if (result.Up < diff)
                    {
                        result.Up = diff;
                    }
                }

            }
            return result;
        }

        private double CalculatePercentChange(double current, double next)
        {
            var diff = next / current;
            return (diff < 1) ? 100 * (1 - diff) : 100 * (diff - 1);
        }

    }
}
