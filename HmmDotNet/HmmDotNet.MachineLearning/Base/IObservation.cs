namespace HmmDotNet.MachineLearning.Base
{
    /// <summary>
    ///     Observation
    /// </summary>
    public interface IObservation
    {
        /// <summary>
        ///     Obsrevation description
        /// </summary>
        string Description { get;}
        /// <summary>
        ///     Observation values
        /// </summary>
        double[] Value { get; }
        /// <summary>
        ///     Number of deimenstions in the observation value
        /// </summary>
        int Dimention { get; }
    }
}
