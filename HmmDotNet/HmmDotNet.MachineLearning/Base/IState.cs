namespace HmmDotNet.MachineLearning.Base
{
    public interface IState
    {
        /// <summary>
        ///     State desctiption
        /// </summary>
        string Description { get; }
        /// <summary>
        ///     State index in the transition array
        /// </summary>
        int Index { get; }
    }
}
