namespace HmmDotNet.MachineLearning.Algorithms
{
    public struct ViterbiDataNode
    {
        public double Value { get; set; }
        public int MaximizingBackTrackState { get; set; }
    }
}
