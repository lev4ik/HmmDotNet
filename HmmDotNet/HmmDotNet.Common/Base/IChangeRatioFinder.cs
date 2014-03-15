using HmmDotNet.Extentions.Data;

namespace HmmDotNet.Extentions.Base
{
    public interface IChangeRatioFinder
    {
        MaximumChangeRatios GetMaximumChangeRatios(double[][] array);
    }
}
